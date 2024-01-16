import torch
import torch.nn as nn
import torch.nn.functional as F

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)

def kl_un(netC, netT, netUL, optimizer, train_dl):
    print(" unlearning:")
    netC.train()

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizer.zero_grad()
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        with torch.no_grad():
            full_teacher_logits = netT(inputs)
            unlearn_teacher_logits = netUL(inputs)
        output = netC(inputs)
        optimizer.zero_grad()
        loss = UnlearnerLoss(output=output, labels=targets, full_teacher_logits=full_teacher_logits,
                             unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=1)
        loss.backward()
        optimizer.step()


def get_feature(netC, dataset, logits=False):
    #print(" Eval:")
    netC.eval()
    feature = []
    #test_dl = dataset
    test_dl = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False)
    for batch_idx, (inputs,   _) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs = inputs.to('cuda')
            if logits:
                preds = netC(inputs)
            else:
                preds = netC.get_feature(inputs)
            feature.append(preds)

    feature = torch.cat(feature, dim=0)
    #feature = np.array(feature)

    return feature




def eval_forget(netC, test_dl, opt,):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_preds_logits = torch.zeros(opt.num_classes).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        #print(targets)
        with torch.no_grad():
            inputs, _ = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs
            targets = (torch.ones_like(targets) * opt.forget_threshold).to(opt.device)
            #print(targets)

            # Evaluate Clean
            preds = netC(inputs)
            logits = F.softmax(preds,dim=1)
            #print(logits)
            #print(targets)
            #print(logits)
            #print(torch.max(logits, 1)[0])
            #print(torch.max(logits, 1)[0].shape)

            total_clean_correct += torch.sum(torch.max(logits, 1)[0] < targets)
            #print(total_clean_correct)
            #print(torch.max(logits, 1)[0])

        total_preds_logits += sum([logits[i] for i in range(len(inputs))])

    print("in test dl total sample are {}".format(total_sample))
    total_preds_logits = total_preds_logits / total_sample

    acc_clean = total_clean_correct * 100.0 / total_sample


    info_string = "Forget rate: {:.4f} ".format(
        acc_clean
    )
    print(info_string)
    print(total_preds_logits)



from transformers import ViTModel

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ViT, self).__init__()
        #self.base = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.base = ViTModel.from_pretrained('./VIT/deit-tiny-patch16-224').cuda()
        self.final = nn.Linear(self.base.config.hidden_size, num_classes).cuda()
        self.num_classes = num_classes
        self.relu = nn.ReLU()

    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:,0])

        return logits

    def get_feature(self, pixel_values):

        outputs = self.base(pixel_values=pixel_values)

        return outputs.last_hidden_state[:,0]