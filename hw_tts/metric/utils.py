import editdistance
import torch
# Don't forget to support cases when target_text == ''

def calc_cer(target_text, predicted_text) -> float:
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text == "":
            return 0.0
        return 1.0
    target_text_splitted = target_text.split()
    predicted_text_splitted = predicted_text.split()
    return editdistance.eval(target_text_splitted, predicted_text_splitted) / len(target_text_splitted)

# def calc_snr(est, target):
#     return 20 * np.log10(np.linalg.norm(target) / (np.linalg.norm(target - est) + 1e-6) + 1e-6)

def calc_si_sdr(est, target):
    alpha = (target * est).sum(dim=-1) / torch.linalg.norm(target, dim=-1)**2
    return 20 * torch.log10(torch.linalg.norm(alpha.unsqueeze(1) * target, dim=-1) / (torch.linalg.norm(alpha.unsqueeze(1) * target - est, dim=-1) + 1e-6) + 1e-6)