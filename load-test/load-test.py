import json
import tarfile

input_data = [
    {
        "inputs": "Women will soon be notified about their breast density after a mammogram. Here’s what that means"
    },
    {
        "inputs": "Bright lights detected by NASA lead to a dancing pair of supermassive black holes"
    },
    {
        "inputs": "Trump allies do damage control as Harris rides fresh momentum post debate"
    },
    {
        "inputs": "Ex-Google exec said goal was to 'crush' competition, trial evidence shows"
    }
    
]

def create_json_files(data):
    for i, d in enumerate(data):
        filename = f"input_{i+1}.json"
        with open(filename, 'w') as f:
            json.dump(d, f, indent=4)

def create_tar_file(input_files, output_filename='inputs.tar.gz'):
    with tarfile.open(output_filename, 'w:gz') as tar:
        for f in input_files:
            tar.add(f)
            
def main():
    create_json_files(input_data)
    input_files = [f'input_{i+1}.json' for i in range(len(input_data))]
    create_tar_file(input_files)
    
    
if __name__ == '__main__':
    main()