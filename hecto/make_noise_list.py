import os
import glob

noisy_folder = "noisy_samples"
train_root = "data/train"

# 1. noisy_samples 폴더 내 파일명 출력
noisy_files = set(os.path.basename(p) for p in glob.glob(os.path.join(noisy_folder, '*')))
print("noisy_samples 파일명:", noisy_files)

# 2. train 전체 이미지 경로 & 파일명 일부 출력
print("\ntrain 전체 내 파일명 (최대 10개):")
cnt = 0
for root, dirs, files in os.walk(train_root):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # 확장자 제한
            print(file)
            cnt += 1
            if cnt > 10:
                break
    if cnt > 10:
        break

# 3. noisy_samples와 train이 매칭되는 파일이 있는지 체크
noise_set = set()
for root, dirs, files in os.walk(train_root):
    for file in files:
        if file in noisy_files:
            noise_set.add(os.path.join(root, file))
print(f"\n매칭된 노이즈 파일 {len(noise_set)}개:", list(noise_set)[:10])

# 4. noise_list.txt로 저장
with open('noise_list.txt', 'w') as f:
    for path in sorted(noise_set):
        f.write(path + '\n')
print(f"\nnoise_list.txt 저장 완료, 총 {len(noise_set)}개")
