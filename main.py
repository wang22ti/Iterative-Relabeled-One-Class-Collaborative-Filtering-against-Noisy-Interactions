from torch.utils.data import DataLoader
import time

from data_loader import *
from evaluate import *
from model import *


def train_model(model, train_data, preference_test_data, interaction_test_data, args, if_print=False):
    best_iter = 0
    if args['dataset'] == 'netflix':
        user_for_test = train_data.all_users[:args['num_user_for_test']]
    else:
        user_for_test = train_data.all_users
    best_interaction_metrics = evaluate(model, user_for_test, interaction_test_data, args['k_list'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)

    for iter in range(args['n_iters']):
        avg_loss = 0
        for idx, (u, i, j) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(u, i, j)
            avg_loss += float(loss)
            loss.backward()
            optimizer.step()

        interaction_metrics = evaluate(model, user_for_test, interaction_test_data, args['k_list'])
        if if_print:
            print(iter, time.time(), avg_loss / (idx + 1), interaction_metrics, sep='\t')
            preference_metrics = evaluate(model, train_data.all_users, preference_test_data, args['k_list'])
            print('[preference]\t', preference_metrics)

        if interaction_metrics[0] > best_interaction_metrics[0]:
            best_iter = iter
            best_interaction_metrics = interaction_metrics
            torch.save(model, args['model_path'])

    print('[best]', time.time(), best_iter, best_interaction_metrics)
    return best_interaction_metrics

args = {
    'dataset': 'ml100k',
    'dim': 16,
    'lr': 1e-4,
    'weight_decay': 0.000025,
    'batch_size': 128,
    'n_epochs': 30,
    'n_iters': 200,
    'k_list': [1, 5, 10],
    'model': 'bpr',
    'pos_rate': 0.35,
    'neg_rate': 0.65,
    'plus': 6,
    'lamda': 1,
    'num_neg_for_test': 1000,
    'num_user_for_test': 1000,
    'seed': 0
}
print(args)
args['k_list'] = [1, 5, 10]

SEED = args['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)  # cpu
torch.cuda.manual_seed(SEED)  # gpu

train_data = TrainData(args['dataset'], args['pos_rate'], args['neg_rate'])
test_data = TestData(train_data)
interaction_test_data = TestInteractionData(test_data, args)
preference_test_data = TestPreferenceData(test_data)
best2_epoch, best2_interaction_metrics = 0, 0
corresponding_preference_metrics = 0

for epoch in range(args['n_epochs']):
    print('epoch', epoch)
    args['model_path'] = os.path.join('model', '%s_%s_%d.pt') % (args['model'], args['dataset'], epoch)

    model = BPR(train_data.num_user, train_data.num_item, args['dim'])

    best_i_metrics = train_model(model, train_data, preference_test_data, interaction_test_data, args,
                                 if_print=False)
    model = torch.load(args['model_path'])
    preference_metrics = evaluate(model, train_data.all_users, preference_test_data, args['k_list'])
    print('[preference]\t', preference_metrics)
    nni.report_intermediate_result(preference_metrics[0])

    if epoch >= 2 and best_i_metrics[0] > best2_interaction_metrics[0] or epoch == 1:
        best2_epoch = epoch
        best2_interaction_metrics = best_i_metrics
        corresponding_preference_metrics = preference_metrics

    train_data.generate_new_train_data(model.U.detach(), model.V.detach(), pos_rate=args['pos_rate'],
                                       neg_rate=args['neg_rate'], plus=args['plus'], lambd=args['lamda'])
    train_data.print_train_pos_rate()
    train_data.save_train_data(epoch)

    interaction_test_data.update_test_items(train_data.train_items)
    SEED = args['seed']
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # cpu
    torch.cuda.manual_seed(SEED)  # gpu

print('[best preference]', best2_epoch, corresponding_preference_metrics, sep='\t')