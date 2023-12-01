import torch.onnx 
from net import Net
model = Net()

#Function to Convert to ONNX 
def Convert_ONNX(): 
    # set the model to inference mode 
    model.eval() 

    # input_sizeの定義
    input_size = (1, 28, 28)

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, *input_size, requires_grad=True)  # 正しい入力サイズを使用

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "mnist_net.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                       'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 
 
    path = "mnist_net.pth" 
    model.load_state_dict(torch.load(path)) 

    Convert_ONNX()