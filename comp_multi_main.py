import os
from PIL import Image
import pandas as pd
import cv2
from EP import comp_EP
from TP import comp_TP
from MSR_CNR_ENL import comp_MSR_CNR_ENL

def list_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

def resize_image_cv2(image_path, size=(300, 300)):
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, size)
    return resized_image

def process_images(image1, image2, rois, background_indices):
    denoised = resize_image_cv2(image1)
    noisy = resize_image_cv2(image2)
    ep=comp_EP(denoised, noisy, rois, background_indices)
    tp=comp_TP(denoised, noisy, rois, background_indices)
    msr,cnr,enl=comp_MSR_CNR_ENL(denoised, noisy, rois, background_indices)
    return {'ep':ep, 'tp':tp, 'msr':msr, 'cnr':cnr, 'enl':enl}

def main(denoised, noisy,result_path,rois,background_indices):
    files = list_image_files(denoised)
    FILENAME=[]
    CNR=[]
    MSR=[]
    TP=[]
    EP=[]
    for file_name in files:
        print(f'processing {file_name}')
        file1_path = os.path.join(denoised, file_name)
        file2_path = os.path.join(noisy, file_name)
        if os.path.exists(file2_path):
            result=process_images(file1_path, file2_path, rois, background_indices)
            FILENAME.append(file_name)
            CNR.append(result['cnr'])
            MSR.append(result['msr'])
            TP.append(result['tp'])
            EP.append(result['ep'])
        else:
            print(f"File {file_name} not found in {noisy}")
        
    data={
        "file_name":FILENAME,
        "CNR":CNR,
        "MSR":MSR,
        "TP":TP,
        "EP":EP
    }
    df=pd.DataFrame(data)
    df.to_csv(result_path,index=False)

if __name__ == "__main__":
    denoised = "D:/projects/IEEE SPS 2024/code/metrics/data_2/denoised" # denoised folder
    noisy = "D:/projects/IEEE SPS 2024/code/metrics/data_2/noisy" # noisy folder
    result_path="D:/projects/IEEE SPS 2024/code/metrics/result/diffusion_5_6.csv" # result file
    ROIs = [(250,0,50,50), (100, 130, 30, 30), (170, 150, 30, 30), (10, 35, 30, 30), (50,150,50,50), (200,200,50,50)] # ROIs
    background_indices = [0]
    main(denoised, noisy,result_path, ROIs, background_indices)
