clear all
%load('/home/dyq/SZL/cifar10_64bit/binary_feat/16bit/feat16_trn.mat');
%load('/home/dyq/SZL/cifar10_64bit/binary_feat/16bit/feat16_tst.mat'); 
load('J2_feat_train0.mat');
load('J2_feat_test0.mat');

train_output = prob_train;
test_output = prob_test;

%mean_trn = mean(mean(train_output));
%mean_tst = mean(mean(test_output));
mean_trn = mean(train_output);
mean_tst = mean(test_output);
for i = 1:16
bi_trn_feat(:,i) = (train_output(:,i)>mean_trn(i));
bi_tst_feat(:,i) = (test_output(:,i)>mean_tst(i));
end


save bi_trn_feat64.mat bi_trn_feat
save bi_tst_feat64.mat bi_tst_feat
