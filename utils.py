import torch


def get_artist_label(midi_data, file_path):
    # Extract the name of the parent folder (artist name)
    artist_name = file_path.parent.name
    
    # Use a dictionary to store artist names and corresponding numerical IDs
    if not hasattr(get_artist_label, 'artist_id_mapping'):
        get_artist_label.artist_id_mapping = {}  # Initialize the dictionary if not exists
    
    # Check if the artist is already in the mapping, if not, assign a new index
    if artist_name not in get_artist_label.artist_id_mapping:
        new_index = len(get_artist_label.artist_id_mapping)
        get_artist_label.artist_id_mapping[artist_name] = new_index
    
    # Return the numerical ID corresponding to the artist
    return get_artist_label.artist_id_mapping[artist_name]

def decode_artist_label(label_id, artist_id_mapping):
    label_id = int(label_id)
    # Reverse the artist_id_mapping dictionary to get artist names from numerical IDs
    reversed_mapping = {v: k for k, v in artist_id_mapping.items()}
    # Return the artist name corresponding to the label_id
    return reversed_mapping[label_id]

def get_batch(data, batch_size):
    indices = torch.randint(len(data), size=(batch_size,))
    batch = [data[i] for i in indices]

    x = torch.stack([value['input_ids'][:-1] for value in batch])
    y = torch.stack([value['input_ids'][1:] for value in batch])

    return x, y

def process_dataset_for_z(data):

    x = torch.stack([value['input_ids'] for value in data])
    lables = [ decode_artist_label(label['labels'],get_artist_label.artist_id_mapping) for label in data] 

    return x, lables

def calculate_feature_pointers(z, y_list_train):
    # Calculate average z_mu and z_log_var

    z = z.float()

    avg_z = torch.mean(z, dim=0)

    # Initialize a dictionary to store feature pointers
    feature_pointers_dict = {}

    # Iterate over values in y_list_train
    for label_value in y_list_train:

        # Create a boolean mask based on the label value
        mask = [label == label_value for label in y_list_train]

        # Filter z_mu based on the mask
        feature_z = z[mask]

        # Calculate average feature z_mu
        feature_avg_z = torch.mean(feature_z, dim=0)

        # Calculate feature pointers
        feature_pointer = feature_avg_z - avg_z

        # Store the feature pointers in the dictionary using the tuple as the key
        feature_pointers_dict[label_value] = feature_pointer

    return feature_pointers_dict