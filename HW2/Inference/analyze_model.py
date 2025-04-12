import torch
import argparse
import os
from prettytable import PrettyTable

def count_parameters(model):
    """Calculate the number of parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_size(num_bytes):
    """Format the byte size into a human-readable format (KB/MB)"""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024**2:
        return f"{num_bytes/1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes/1024**2:.2f} MB"
    else:
        return f"{num_bytes/1024**3:.2f} GB"

def estimate_model_size(model):
    """Estimate the size of the model in memory"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return param_size + buffer_size

def print_model_layers(model):
    """Print the parameter statistics of each layer in the model"""
    table = PrettyTable(["Layer Name", "Shape", "Parameters", "Trainable"])
    table.align = "l"
    
    for name, param in model.named_parameters():
        table.add_row([
            name, 
            str(list(param.shape)), 
            param.numel(), 
            "Yes" if param.requires_grad else "No"
        ])
    
    print(table)

def analyze_model(model_path):
    """Analyze a saved model file"""
    print(f"Loading model from: {model_path}")
    
    # Load the model
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if the loaded file is a state_dict
        if isinstance(model, dict) and 'state_dict' in model:
            print("Detected a checkpoint dictionary, extracting state_dict...")
            model = model['state_dict']
            print("Note: Only analyzing state_dict, not the full model architecture.")
            print("Some information might be limited.")
            
        # If it's a state_dict, the model architecture cannot be directly analyzed
        if isinstance(model, dict):
            # Analyze the state_dict
            total_params = sum(p.numel() for p in model.values())
            # Assume all parameters are trainable since it's unknown
            print(f"\nTotal Parameters: {total_params:,}")
            
            # Estimate model size
            param_size = sum(p.nelement() * p.element_size() for p in model.values())
            print(f"Estimated Model Size: {format_size(param_size)}")
            
            # Display parameter distribution
            table = PrettyTable(["Layer Name", "Shape", "Parameters"])
            table.align = "l"
            
            for name, param in model.items():
                table.add_row([name, str(list(param.shape)), param.numel()])
            
            print("\nParameter Distribution:")
            print(table)
            
        else:
            # If it's a full model object, perform a more detailed analysis
            total_params, trainable_params = count_parameters(model)
            
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            print(f"Frozen Parameters: {total_params - trainable_params:,}")
            
            # Estimate model size
            model_size = estimate_model_size(model)
            print(f"Estimated Model Size: {format_size(model_size)}")
            
            # Display model architecture
            print("\nModel Architecture:")
            print(model)
            
            # Display parameter distribution
            print("\nParameter Distribution:")
            print_model_layers(model)
    
    except Exception as e:
        print(f"Error loading model: {e}")
        
        # Attempt to load as state_dict
        try:
            print("Attempting to load as state_dict...")
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            if isinstance(state_dict, dict):
                # Calculate total parameters
                total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                
                print(f"\nTotal Parameters: {total_params:,}")
                
                # Display parameter distribution
                table = PrettyTable(["Layer Name", "Shape", "Parameters"])
                table.align = "l"
                
                for name, param in state_dict.items():
                    if isinstance(param, torch.Tensor):
                        table.add_row([name, str(list(param.shape)), param.numel()])
                
                print("\nParameter Distribution:")
                print(table)
            else:
                print("Unable to parse model format")
        except Exception as inner_e:
            print(f"Unable to analyze model: {inner_e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch model parameters")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.pt, .pth)")
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file not found {args.model}")
        return
    
    analyze_model(args.model)

if __name__ == "__main__":
    try:
        import prettytable
    except ImportError:
        print("prettytable library not installed, installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "prettytable"])
        print("Installation complete!")
    
    main()