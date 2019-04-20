import json
import numpy as np
dict = {
    'results':[]
}

def json_write_to_json(result, json_file_name):
    for iter in range(len(result)):
        rects = []
        for i in range(len(result[iter]['ann']['bboxes'])):
            if  float(result[iter]['ann']['confidences'][i]) >= 0.1:
                rect = {
                    'xmin': int(round(result[iter]['ann']['bboxes'][i][0])),
                    'xmax': int(round(result[iter]['ann']['bboxes'][i][0] + result[iter]['ann']['bboxes'][i][2] - 1)),
                    'ymin': int(round(result[iter]['ann']['bboxes'][i][1])),
                    'ymax': int(round(result[iter]['ann']['bboxes'][i][1] + result[iter]['ann']['bboxes'][i][3] - 1)),
                    'label': int(result[iter]['ann']['labels'][i] + 1),
                    'confidence': float(round(result[iter]['ann']['confidences'][i], 1)),
                }
                rects.append(rect)

        task = {
            'filename': result[iter]['filename'],
            'rects': rects,
        }
        dict['results'].append(task)
    with open(json_file_name, 'w') as outfile:
        json.dump(dict, outfile)

if __name__ == '__main__':
    file_name = './results.pkl.json'
    json_data = open(file_name).read()
    print(len(json_data))
    #result = json.loads(json_data)
    #json_write_to_json(result, 'final.json')