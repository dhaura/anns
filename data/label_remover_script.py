def convert_comma_to_space(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    with open(output_file, 'w') as outfile:
        for line in lines:
            # Split the line by commas.
            items = line.strip().split(',')
            # Remove the last item.
            items = items[:-1]
            # Join the items with spaces and write to the output file.
            outfile.write(' '.join(items) + '\n')

input_file = input("Enter the input file path (e.g., iris.data): ")
output_file = input("Enter the output file path (e.g., iris_converted.data): ")
convert_comma_to_space(input_file, output_file)

print(f"Converted {input_file} to {output_file} with spaces instead of commas.")
