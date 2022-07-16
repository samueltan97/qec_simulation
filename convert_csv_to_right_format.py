import pandas as pd

lines = open('results.csv', 'r').readlines()
final_index_list = []
counter = 0
for index, line in enumerate(lines[1:]):
    if int(line.split(',')[0]) == 55:
        final_index_list.append(index)
        # for row in lines[index - 55:index + 1]:
        #     counter += 1
        #     final_list.append(row)
print(final_index_list)
final_list = []
for index in final_index_list:
    final_list += lines[index-54:index+2]
    # print(lines[index-55:index+1])

new_final_list = final_list[:56]
for index, line in enumerate(final_list[56:]):
    print(index, len(final_list[56:]) - 1)
    if index >= (len(final_list[56:]) - 56):
        new_logical_error_rate = str((float(new_final_list[index % 56].strip().split(',')[-1]) + float(line.strip().split(',')[-1]))/99) + '\n'
    else:
        new_logical_error_rate = str(float(new_final_list[index % 56].strip().split(',')[-1]) + float(line.strip().split(',')[-1])) + '\n'
    print(new_logical_error_rate)
    new_final_list[index % 56] = ','.join(new_final_list[index % 56].strip().split(',')[:-1] + [new_logical_error_rate])

new_file = open('processed_results.csv', 'w')
new_file.writelines(new_final_list)
new_file.close()
# print(counter)

final_list = [lines[0]]


