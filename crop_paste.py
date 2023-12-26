import cv2, os , random
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

## Parameters
CFG = {
    'base_image_path': '../wim_data/SAM_2_objects/conveyer_resized.png', # 배경 컨베이어밸트 이미지
    'object_images_folder': '../wim_data/objects/images_object2/', # 객체 이미지 경로 
    'max_objects' : 12,  # 배경 이미지에 붙일 최대 객체 수
    'rectangle' : (300, 0, 680, 740), # 배경의 지정된 영역 설정 (x 시작, y 시작, 너비, 높이)
    'max_overlap': 0.3, # 객체들의 겹침 허용 비율
    'num_iter': 3, # 생성할 이미지 수
    'max_dim': 500, # 샘플링 객체 크기 제한
    'output_folder': '../wim_data/crop_paste/', # 생성 이미지 저장 경로
}

# YOLO 레이블 변환
def convert_to_yolo_label(x, y, w, h, img_width, img_height):
    # 중심 좌표 및 너비, 높이를 이미지 크기로 정규화
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

# 겹치는 비율 확인
def overlap_ratio(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 겹치는 영역의 좌표와 크기를 계산
    x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    y_overlap = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap

    # 두 영역의 최소 넓이를 계산
    area1, area2 = w1*h1, w2*h2
    min_area = min(area1, area2)

    # 겹치는 영역의 비율을 계산
    if min_area == 0:
        return 0
    overlap_ratio = overlap_area / float(min_area)

    return overlap_ratio

# 겹치지 않는 위치를 찾는 함수
def find_overlapping_position(obj_shape, existing_objects, rectangle, max_overlap=0.5):
    max_attempts = 100  # 겹치지 않는 위치를 찾기 위한 최대 시도 횟수
    obj_height, obj_width = obj_shape  # 객체의 높이와 너비

    for _ in range(max_attempts):
        # rectangle 영역 내에서 랜덤 위치를 생성
        
        if rectangle[2] > obj_width and rectangle[3] > obj_height:
            x_offset = random.randint(rectangle[0], rectangle[0] + rectangle[2] - obj_width)
            y_offset = random.randint(rectangle[1], rectangle[1] + rectangle[3] - obj_height)
            new_obj = (x_offset, y_offset, obj_width, obj_height)

            # 겹침이 허용된 범위 내인지 검사
            if all(overlap_ratio(new_obj, existing_obj) < max_overlap for existing_obj in existing_objects):
                return new_obj
        else:
            # The object is too large to fit within the rectangle, so skip it
            return None

    return None  # 겹치지 않는 위치를 찾지 못한 경우

# 객체의 크기를 계산하는 함수
def calculate_object_size(obj_path, obj_name):
    obj_img_path = os.path.join(obj_path, obj_name)
    image = cv2.imread(obj_img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return 0  # 이미지를 불러오지 못한 경우
    return image.shape[0] * image.shape[1]  # height * width

def get_image_dimensions(obj_path, obj_name):
    obj_img_path = os.path.join(obj_path, obj_name)
    image = cv2.imread(obj_img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return 0, 0  # 이미지를 불러오지 못한 경우
    return image.shape[1], image.shape[0]  # width, height

# 객체 이미지 목록 가져오기
object_images = [f for f in os.listdir(CFG['object_images_folder']) if f.endswith('.jpg') or f.endswith('.png')]

## dim 512 보다 큰 이미지는 제외 
filtered_objects = [obj for obj in object_images if all(dim <= CFG['max_dim'] for dim in get_image_dimensions(CFG['object_images_folder'], obj))]
print(f'총 객체 수 : {len(filtered_objects)}')


work_date = datetime.now().strftime("%Y%m%d%H%M")

# num_iter (생성할 이미지 수)
for i in tqdm(range(CFG['num_iter'])):
    # 기본 이미지 로드
    base_image = cv2.imread(CFG['base_image_path'])
    # 객체 이미지 배치 및 라벨 생성, 객체 클래스, 배치된 위치 저장
    labels = []
    classes = []
    placed_objects = []
    
    # 객체 샘플링
    selected_objects = random.sample(filtered_objects, CFG['max_objects'])

    # 객체의 크기가 큰 순서대로 정렬하기 위한 객체 크기 확인 (딕셔너리 컴프래헨션 구조)
    object_sizes = {obj: calculate_object_size(CFG['object_images_folder'],obj) for obj in selected_objects}
    # 크기에 따라 selected_objects 정렬 (내림차순)
    selected_objects_sorted = sorted(selected_objects, key=lambda obj: object_sizes[obj], reverse=True)
    print(selected_objects_sorted)
    # max_objects의 수 만큼 반복
    for obj_img_name in selected_objects_sorted:
        parts = obj_img_name.split('_')
        obj_class = parts[2]
        
        obj_img_path = os.path.join(CFG['object_images_folder'], obj_img_name)
        obj_img = cv2.imread(obj_img_path, -1)
        if obj_img is None:
            print(f"Image at {obj_img_path} could not be loaded.")
            continue
        
        # 알파 채널과 BGR 채널 분리
        alpha_channel = obj_img[:, :, 3]  # 알파 채널 추출
        obj_img = obj_img[:, :, :3]  # BGR 채널만 추출
        
        # 알파 채널을 마스크로 사용해 배경을 제거
        mask = alpha_channel / 255.0  # 정규화된 마스크
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # 마스크를 3채널(BGR)로 확장
        
        obj_img_float = obj_img.astype(float)  # obj_img를 생성된 마스크와 연산을 위해 부동소수점타입으로 변경
        obj_img = cv2.multiply(obj_img_float, mask_3d)
        
        # 마스킹된 이미지의 타입을 표준 이미지 타입인 uint8로 변환
        obj_img = cv2.convertScaleAbs(obj_img)
        
        # 객체의 크기를 가져옴
        obj_height, obj_width = obj_img.shape[:2]
        
        # 겹치지 않는 위치 찾기
        position = find_overlapping_position((obj_height, obj_width), placed_objects, CFG['rectangle'], CFG['max_overlap'])
        if position is None:
            continue
        
        
        # 객체를 배경 이미지에 붙임
        x_offset, y_offset, _, _ = position
        y1, y2 = y_offset, y_offset + obj_height
        x1, x2 = x_offset, x_offset + obj_width
        alpha_s = mask.astype(float)
        alpha_l = 1.0 - alpha_s
        
        
        # 배경 이미지의 영역을 넘어가는 것을 방지 객체의 왼쪽 상단 좌표 확인
        y1 = max(0, y1)
        y2 = min(base_image.shape[0], y2)
        x1 = max(0, x1)
        x2 = min(base_image.shape[1], x2)

        # 배경 이미지 영역 제한에 의해 변경된 좌표에 맞춰 width, height 재설정
        slice_height = y2 - y1
        slice_width = x2 - x1

        # 재설정 되었다면 객체, 알파 채널 리사이즈
        resize_needed = slice_height != obj_height or slice_width != obj_width
        if resize_needed:
            obj_img = cv2.resize(obj_img, (slice_width, slice_height))
            alpha_s = cv2.resize(alpha_s, (slice_width, slice_height))
            alpha_l = 1.0 - alpha_s
        
        
        # 알파 블렌딩을 사용하여 이미지를 합성합니다.
        for c in range(3): # BGR 채널 수
            base_image[y1:y2, x1:x2, c] = alpha_s * obj_img[:, :, c] + alpha_l * base_image[y1:y2, x1:x2, c]
        
        # 객체의 위치, 크기 모두 저장
        if resize_needed:
            # 변경된 크기
            placed_objects.append((x_offset, y_offset, slice_width, slice_height))
        else:
            # 기존 크기 그대로
            placed_objects.append((x_offset, y_offset, obj_width, obj_height))
        # 클래스 저장
        classes.append(obj_class)
    
    # 결과 이미지, 레이블 저장
    task_name = f"{work_date}_{str(CFG['max_objects'])}_{str(CFG['num_iter'])}"
    dir_name = os.path.join(CFG['output_folder'], task_name)
    save_name = f'paste_{len(classes)}_{i}'
    
    
    output_image_path = os.path.join(dir_name, 'images')
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
        
    output_label_path = os.path.join(dir_name, 'labels')
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)
        
    output_image_path = os.path.join(output_image_path ,save_name +'.png')
    output_label_path = os.path.join(output_label_path, save_name +'.txt')
    
    
    cv2.imwrite(output_image_path, base_image)
    
    with open(output_label_path, 'w') as file:
        # 라벨 생성
        for j,position in enumerate(placed_objects):
            x_offset, y_offset, obj_width, obj_height = position
            # 라벨 좌표 및 바운딩 박스 크기를 계산
            yolo_label = convert_to_yolo_label(x_offset, y_offset, obj_width, obj_height, base_image.shape[1], base_image.shape[0])
            file.write(f"{classes[j]} {yolo_label[0]} {yolo_label[1]} {yolo_label[2]} {yolo_label[3]}" + "\n")
        