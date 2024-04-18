from utils.config_parser import parse_config
from argparse import ArgumentParser
import torch
import wandb
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, auc
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np


def get_model(ckpt, extra_feats, config):
    # Set device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if extra_feats:
        from utils.light_modules import LightningKWT_extrafeats
        model = LightningKWT_extrafeats.load_from_checkpoint(ckpt, config=config)
        
    else:
        from utils.light_modules import LightningKWT
        model = LightningKWT.load_from_checkpoint(ckpt, config=config)
        
    model.to(device)
    return model, device

def get_dataloader(config, extra_feats):

    from src.data.dataset import SpectrogramDataset
    test_set = SpectrogramDataset(manifest_path=config['eval_manifest_path'],labels_map=config['labels_map'],audio_config=config['audio_config'], augment=False)
    
   
    #all classes in the FSD50K dataset - 200 
    classes = {"Accelerating_and_revving_and_vroom": 0, "Accordion": 1, "Acoustic_guitar": 2, "Aircraft": 3, "Alarm": 4, "Animal": 5, "Applause": 6, "Bark": 7, "Bass_drum": 8, "Bass_guitar": 9, "Bathtub_(filling_or_washing)": 10, "Bell": 11, "Bicycle": 12, "Bicycle_bell": 13, "Bird": 14, "Bird_vocalization_and_bird_call_and_bird_song": 15, "Boat_and_Water_vehicle": 16, "Boiling": 17, "Boom": 18, "Bowed_string_instrument": 19, "Brass_instrument": 20, "Breathing": 21, "Burping_and_eructation": 22, "Bus": 23, "Buzz": 24, "Camera": 25, "Car": 26, "Car_passing_by": 27, "Cat": 28, "Chatter": 29, "Cheering": 30, "Chewing_and_mastication": 31, "Chicken_and_rooster": 32, "Child_speech_and_kid_speaking": 33, "Chime": 34, "Chink_and_clink": 35, "Chirp_and_tweet": 36, "Chuckle_and_chortle": 37, "Church_bell": 38, "Clapping": 39, "Clock": 40, "Coin_(dropping)": 41, "Computer_keyboard": 42, "Conversation": 43, "Cough": 44, "Cowbell": 45, "Crack": 46, "Crackle": 47, "Crash_cymbal": 48, "Cricket": 49, "Crow": 50, "Crowd": 51, "Crumpling_and_crinkling": 52, "Crushing": 53, "Crying_and_sobbing": 54, "Cupboard_open_or_close": 55, "Cutlery_and_silverware": 56, "Cymbal": 57, "Dishes_and_pots_and_pans": 58, "Dog": 59, "Domestic_animals_and_pets": 60, "Domestic_sounds_and_home_sounds": 61, "Door": 62, "Doorbell": 63, "Drawer_open_or_close": 64, "Drill": 65, "Drip": 66, "Drum": 67, "Drum_kit": 68, "Electric_guitar": 69, "Engine": 70, "Engine_starting": 71, "Explosion": 72, "Fart": 73, "Female_singing": 74, "Female_speech_and_woman_speaking": 75, "Fill_(with_liquid)": 76, "Finger_snapping": 77, "Fire": 78, "Fireworks": 79, "Fixed-wing_aircraft_and_airplane": 80, "Fowl": 81, "Frog": 82, "Frying_(food)": 83, "Gasp": 84, "Giggle": 85, "Glass": 86, "Glockenspiel": 87, "Gong": 88, "Growling": 89, "Guitar": 90, "Gull_and_seagull": 91, "Gunshot_and_gunfire": 92, "Gurgling": 93, "Hammer": 94, "Hands": 95, "Harmonica": 96, "Harp": 97, "Hi-hat": 98, "Hiss": 99, "Human_group_actions": 100, "Human_voice": 101, "Idling": 102, "Insect": 103, "Keyboard_(musical)": 104, "Keys_jangling": 105, "Knock": 106, "Laughter": 107, "Liquid": 108, "Livestock_and_farm_animals_and_working_animals": 109, "Male_singing": 110, "Male_speech_and_man_speaking": 111, "Mallet_percussion": 112, "Marimba_and_xylophone": 113, "Mechanical_fan": 114, "Mechanisms": 115, "Meow": 116, "Microwave_oven": 117, "Motor_vehicle_(road)": 118, "Motorcycle": 119, "Music": 120, "Musical_instrument": 121, "Ocean": 122, "Organ": 123, "Packing_tape_and_duct_tape": 124, "Percussion": 125, "Piano": 126, "Plucked_string_instrument": 127, "Pour": 128, "Power_tool": 129, "Printer": 130, "Purr": 131, "Race_car_and_auto_racing": 132, "Rail_transport": 133, "Rain": 134, "Raindrop": 135, "Ratchet_and_pawl": 136, "Rattle": 137, "Rattle_(instrument)": 138, "Respiratory_sounds": 139, "Ringtone": 140, "Run": 141, "Sawing": 142, "Scissors": 143, "Scratching_(performance_technique)": 144, "Screaming": 145, "Screech": 146, "Shatter": 147, "Shout": 148, "Sigh": 149, "Singing": 150, "Sink_(filling_or_washing)": 151, "Siren": 152, "Skateboard": 153, "Slam": 154, "Sliding_door": 155, "Snare_drum": 156, "Sneeze": 157, "Speech": 158, "Speech_synthesizer": 159, "Splash_and_splatter": 160, "Squeak": 161, "Stream": 162, "Strum": 163, "Subway_and_metro_and_underground": 164, "Tabla": 165, "Tambourine": 166, "Tap": 167, "Tearing": 168, "Telephone": 169, "Thump_and_thud": 170, "Thunder": 171, "Thunderstorm": 172, "Tick": 173, "Tick-tock": 174, "Toilet_flush": 175, "Tools": 176, "Traffic_noise_and_roadway_noise": 177, "Train": 178, "Trickle_and_dribble": 179, "Truck": 180, "Trumpet": 181, "Typewriter": 182, "Typing": 183, "Vehicle": 184, "Vehicle_horn_and_car_horn_and_honking": 185, "Walk_and_footsteps": 186, "Water": 187, "Water_tap_and_faucet": 188, "Waves_and_surf": 189, "Whispering": 190, "Whoosh_and_swoosh_and_swish": 191, "Wild_animals": 192, "Wind": 193, "Wind_chime": 194, "Wind_instrument_and_woodwind_instrument": 195, "Wood": 196, "Writing": 197, "Yell": 198, "Zipper_(clothing)": 199}

    test_loader = DataLoader(test_set, batch_size=50)
    
    return test_loader, classes

def get_predictions(device, model, test_loader):
    predicted_labels = torch.tensor([]).to(device)
    true_labels = torch.tensor([]).to(device)

    for specs, labels in tqdm(test_loader):
        specs = specs.to(device)  #sending spectograms to device
        labels = labels.to(device) # sending labels to device
        output = model(specs)  # feeding data to network
        predicted_labels = torch.cat((predicted_labels, output.argmax(1))) #argmax to find "hot one" in one hot encoding
        true_labels = torch.cat((true_labels, labels.argmax(1))) #argmax to find "hot one" in one hot encoding
        
    predicted_labels = torch.flatten(predicted_labels).cpu() #flattening dimensions
    true_labels = torch.flatten(true_labels).cpu() #flattening dimensions
    return predicted_labels, true_labels

def test_pipeline(model, test_loader, device, classes, extra_feats):
    print("Initiating testing...")
    
    predicted_labels, true_labels = get_predictions(device, model, test_loader)

    # Creating confusion matrix using sklearn function 
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting confusion matrix
    df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    confusion_matrix_fig = plt.figure(figsize = (12,7))
    plt.title("Confusion Matrix")
    sns.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig('confusion_matrix.png')
    plt.show()


    # Precision and Recall
    precision_per_class = []
    recall_per_class = []

    
    # finding precision adn recall for each class with sklearn 
    for i in range(len(classes)):
        precision = precision_score(true_labels == i, predicted_labels == i)
        recall = recall_score(true_labels == i, predicted_labels == i)
        precision_per_class.append(precision)
        recall_per_class.append(recall)

    print("Precision of class 'Covid': ", precision_per_class[0])
    print("Precision of class 'healthy': ", precision_per_class[1])
    print("Precision of class 'symptomatic': ", precision_per_class[2])
    
    print("Recall of class 'Covid': ", recall_per_class[0])
    print("Recall of class 'healthy': ", recall_per_class[1])
    print("Recall of class 'symptomatic': ", recall_per_class[2])


    #Plotting precision-recall curve

    #The precision-recall curve shows the tradeoff between precision and recall for different threshold.
    #A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate,
    # and high recall relates to a low false negative rate. (from sklearn documentation) --> so we want both high recall and high precision for a good model

    precision_recall_fig = plt.figure(figsize=(12, 7))

    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(true_labels == i, predicted_labels == i)
        class_name = list(classes.keys())[list(classes.values()).index(i)]
        plt.plot(recall, precision, label=' {} (Precision = {:.2f}, Recall = {:.2f})'.format(class_name, precision_per_class[i], recall_per_class[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True) 
    plt.savefig('precision_recall_curve.png')
    plt.show()
    
    # Plotting ROC curve

    #ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis. 
    #This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. 
    #This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.
    #The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.  (from sklearn documentation)

    ROC_curve_fig = plt.figure(figsize=(12, 7))
    #for multi-class classification, it is necessary to binarize the output. One ROC curve can be drawn per label.
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(true_labels == i, predicted_labels == i)
        roc_auc = auc(fpr, tpr)
        class_name = list(classes.keys())[list(classes.values()).index(i)]
        plt.plot(fpr, tpr, lw=2, label='ROC curve for {} (AUC = {:.2f})'.format(class_name, roc_auc))


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #ROC_fig = plt.figure(figsize = (12,7))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive (FP) Rate')
    plt.ylabel('True Positive (TP) Rate')
    plt.title('ROC Curve')
    plt.legend(loc="best")
    plt.grid(True) 
    plt.savefig('ROC_curve.png')
    plt.show()

    #The ROC curve is a plot of the true positive rate (TP) against the false positive rate (FP) at various threshold settings. 
    #Each point on the ROC curve represents a sensitivity/specificity pair corresponding to a particular decision threshold.
    # The AUC measures the entire two-dimensional area underneath the ROC curve from (0,0) to (1,1).

    #Interpreting the AUC:
   # AUC = 1: Perfect classifier. It means the model has perfect discrimination ability, achieving a TPR of 1 (sensitivity) while maintaining an FPR of 0 (specificity) across all thresholds.
    #AUC = 0.5: Random classifier. It indicates that the model has no discrimination ability better than random guessing. The ROC curve coincides with the diagonal line (the line of no-discrimination).
    #AUC < 0.5: Worse than random classifier. It suggests that the model's predictions are inversely related to the true labels, indicating a model with poor performance.

    #logging to    #logging confusion matrix, precision-recall curve, ROC-curve to wandb
    wandb.log({'confusion_matrix.png' : wandb.Image(confusion_matrix_fig)}) 
    # Logging precision-recall curve figure
    wandb.log({'precision_recall_curve.png': wandb.Image(precision_recall_fig)})
    # Logging ROC curve figure
    wandb.log({'ROC_curve.png': wandb.Image(ROC_curve_fig)})

def main(args):
    """
    Loads best checkpoint from training run
    Runs test loop using the test data 
    Plots confusion matrix, precision-recall curve and ROC-curve over results
    """

    config = parse_config(args.conf)
    model, device = get_model(args.ckpt, args.extra_feats, config)
    test_loader, classes = get_dataloader(config, args.extra_feats)

    # Logging setup
    if args.id:
        config["exp"]["exp_name"] = config["exp"]["exp_name"] + args.id

    if config["exp"]["wandb"]:
        wandb.login()
        with wandb.init(project=config["exp"]["proj_name"], name=config["exp"]["exp_name"], config=config["hparams"],entity=config["exp"]["entity"]):
            test_pipeline(model, test_loader, device, classes, args.extra_feats)

    else:
        test_pipeline(model, test_loader, device, classes, args.extra_feats)
    

if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config.yaml file.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file.", default=None)
    parser.add_argument("--id", type=str, required=False, help="Optional experiment identifier.", default=None)
    parser.add_argument("--extra_feats", action='store_true')    
    args = parser.parse_args()

    main(args)
