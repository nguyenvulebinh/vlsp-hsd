def export_result_submit(result_json, file_out, file_template):
    with open(file_out, 'w', encoding='utf-8') as file_out:
        with open(file_template, 'r', encoding='utf-8') as file_in:
            lines = file_in.read().split('\n')
            for line in lines:
                if line.startswith('test_'):
                    key, value = line.split(',')
                    file_out.write("{},{}\n".format(key, result_json[key]))
                else:
                    file_out.write(line + "\n")
