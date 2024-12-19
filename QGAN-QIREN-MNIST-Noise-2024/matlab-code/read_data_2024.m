clc
clear all
close all
% path_head = 'D:\2022AAA-File\code\QGAN-Quantum Generative Adversarial Networks-2023\Unrolled-QGAN-GAN-2024\resutls\[0, 1, 2]_4p_2l_1bs';
% idx = 0:370:7500;%7260:10:7500
% k = 1;
% for i = idx
%     subplot(5,5,k)
%     path_all =  strcat(path_head,'\',num2str(i),'.png');
%     a = imread(path_all);
%     imshow(a(:,:,1));
%     k=k+1;
% end
look_num = 0;
ALL_EXP = 0:10;
path_head = strcat('D:\2022AAA-File\code\QGAN-Quantum Generative Adversarial Networks-2023\Unrolled-QGAN-GAN-2024\resutls\dig_num_[',num2str(look_num),']_patch_num_4_layers_num_2_batch_size_num_1_K_');
path_end = strcat('\fid_[',num2str(look_num),'].npy');
% path_FID = 'D:\2022AAA-File\code\QGAN-Quantum Generative Adversarial Networks-2023\Unrolled-QGAN-GAN-2024\resutls\dig_num_[0]_patch_num_4_layers_num_2_batch_size_num_1\fid_[0].npy';
% path_FID_K_1 = 'D:\2022AAA-File\code\QGAN-Quantum Generative Adversarial Networks-2023\Unrolled-QGAN-GAN-2024\resutls\dig_num_[0]_patch_num_4_layers_num_2_batch_size_num_1_K_1\fid_[0].npy';
% path_FID_K_2 = 'D:\2022AAA-File\code\QGAN-Quantum Generative Adversarial Networks-2023\Unrolled-QGAN-GAN-2024\resutls\dig_num_[0]_patch_num_4_layers_num_2_batch_size_num_1_K_2\fid_[0].npy';
% FID = readNPY(path_FID);
subplot(1,2,1);
for i = ALL_EXP
    path_FID_all = strcat(path_head,num2str(i),path_end);
    FID_K_1 = readNPY(path_FID_all);

    if i==0
        plot(FID_K_1,'LineWidth',2)
    else
        plot(FID_K_1)
        
    end
    hold on
end
legend('FID-K=0','FID-K=1','FID-K=2','FID-K=3','FID-K=4','FID-K=5','FID-K=6','FID-K=7','FID-K=8','FID-K=9','FID-K=10')
title('FID')
% 
% 
% path_KL_end = '\KL_[0].npy';
path_KL_end = strcat('\KL_[',num2str(look_num),'].npy');

subplot(1,2,2);
for i = ALL_EXP
    path_KL_all = strcat(path_head,num2str(i),path_KL_end);
    KL_K_1 = readNPY(path_KL_all);

    if i==0
        plot(KL_K_1,'LineWidth',2)
    else
        plot(KL_K_1)
        
    end
    hold on
end
legend('KL-K=0','KL-K=1','KL-K=2','KL-K=3','KL-K=4','KL-K=5','KL-K=6','KL-K=7','KL-K=8','KL-K=9','KL-K=10')
title('KL')