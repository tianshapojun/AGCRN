# Run.py
elif args.mode == 'Test':
    model.load_state_dict(torch.load('/content/gdrive/MyDrive/Models/AGCRN/experiments/20240227014816/best_model.pth'))
    print("Load saved model")
    data = np.load('../data/Macro/samples.npy')[:,:,0]
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    data_series = data.reshape(-1,12,data.shape[1],data.shape[2])
    print(data_series[0,:,0])
    from lib.dataloader import normalize_dataset
    data_series,_ = normalize_dataset(data_series,'std',args.column_wise)
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    data_series = TensorFloat(data_series)
    output = trainer.test(model, trainer.args, data_series, scaler, trainer.logger)
    print(output[0,:,10])

# ./model/BasicTrainer.py
@staticmethod
    def test(model, args, data_loader, scaler, logger, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():            
            output = model(data_loader, [], teacher_forcing_ratio=0)
        return output
