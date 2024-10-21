import json


def read_coco_json(file_path):
    """
    Read and print the content of a COCO format JSON file.

    Parameters:
    - file_path (str): The path to the COCO JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data.keys())

            annotations = data['annotations']
            for i in range(10):
                print(annotations[i])

        # # 打印整个JSON文件的内容
        # print(json.dumps(data, indent=4))
        #
        # # 如果只想查看特定部分，如所有的images或categories，可以单独打印：
        # print("Images:")
        # for image in data['images']:
        #     print(image)
        #
        # print("Annotations:")
        # for annotation in data['annotations']:
        #     print(annotation)
        #
        # print("Categories:")
        # for category in data['categories']:
        #     print(category)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print("Error: The file could not be decoded.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Replace 'path_to_your_coco_file.json' with the actual path to your COCO JSON file
file_path = 'dataset/interaction_ingame.json'
read_coco_json(file_path)
