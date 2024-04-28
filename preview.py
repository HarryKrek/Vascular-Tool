import yaml
from vascular_tool import worker_process, find_images_in_path
from math import floor
from multiprocessing import Pool, cpu_count, set_start_method


def main(config_path: str, imagePath: str, resultsPath: str) -> None:
    config_path = ".\\config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    full_images = find_images_in_path(imagePath)
    examplePoints = [0, floor((len(full_images) - 1) / 2), -1]
    images = [full_images[0], full_images[examplePoints[1]], full_images[-1]]

    args = [
        (examplePoints[i], image, resultsPath, config) for i, image in enumerate(images)
    ]
    print("Running Example Images")
    print("Please note program will not close without closing all pyplot instances")
    with Pool(3) as p:
        results = p.map(worker_process, args)
        p.close()
    # for arg in args:
    #     worker_process(arg)


if __name__ == "__main__":
    path = "F://20230304_075556_96 wel plate_2D co culture_ HAEC P2_ASC52 P8_20230303_4X_TIME LAPSE//Wellc8//F2"
    config = None
    resultsPath = "./Results/"
    main(config, path, resultsPath)
