import torch
import wandb

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, matthews_corrcoef, jaccard_score


class Evaluation:
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, loss, test_loader, model, device):
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            total_loss = 0
            results = []

            x = torch.tensor([])
            y = torch.tensor([])

            for test_loader_data, labels, item_name in test_loader:
                test_loader_data = test_loader_data.to(device)
                labels = labels.to(device)

                outputs = model(test_loader_data)
                total_loss += loss(outputs, labels).item()

                x = torch.cat((x, outputs.cpu()))
                y = torch.cat((y, labels.cpu()))

                if self.debug:
                    print(
                        f"evaluation item: {item_name}, prediction: {outputs.cpu().detach().numpy().tolist()}, "
                        f"class prediction: {outputs.cpu().detach().numpy().argmax(-1).tolist()}, ground "
                        f"truth: {labels.cpu().detach().numpy().tolist()}")

                results.extend(zip(labels.cpu(), outputs.cpu(), item_name))

            eval_loss = total_loss / len(test_loader)

            class_report = classification_report(
                y.detach().numpy(),
                x.detach().numpy().argmax(-1),
                labels=list(test_loader.dataset.classes.keys()),
                target_names=list(test_loader.dataset.classes.values()),
                output_dict=True,
                zero_division=0.0
            )

            mathews_coeff = matthews_corrcoef(y.detach().numpy(), x.detach().numpy().argmax(-1))
            jacc_score = jaccard_score(y.detach().numpy(), x.detach().numpy().argmax(-1))
            balanced_accuracy = balanced_accuracy_score(y.detach().numpy(), x.detach().numpy().argmax(-1))

            conf_mat = confusion_matrix(
                y.detach().numpy(),
                x.detach().numpy().argmax(-1),
                labels=list(test_loader.dataset.classes.keys())
            )

            column_names = list(test_loader.dataset.classes.values())
            conf_mat = conf_mat.tolist()

            print(
                f"evaluation_report: {dict({'mathews_correlation_coefficient': mathews_coeff, 'jaccard_score': jacc_score, 'balanced_accuracy': balanced_accuracy, **class_report})}," 
                f"evaluation_loss: {eval_loss}, evaluation_confusion_matrix: {conf_mat}"
            )

            if wandb.run is not None:
                for index, row in enumerate(conf_mat):
                    row.insert(0, column_names[index])

                wandb_table = wandb.Table(data=conf_mat, columns=["names (real ↓/predicted →)"] + column_names)

                wandb.log(
                    {f"evaluation_report": {'loss': eval_loss, 'mathews_correlation_coefficient': mathews_coeff, 'jaccard_score': jacc_score, 'balanced_accuracy': balanced_accuracy, **class_report},
                     f"evaluation_confusion_matrix": wandb_table
                     })

            return eval_loss, results
