import torch
import sys
from dciknn_cuda.dciknn_cuda import DCI


def print_without_newline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def get_data_at_level(data, level):
    for key, val in data.items():
        if str(level) in key and 'path' not in key:
            return key, val
    return 'HR', data['HR']


def valid_input_at_level(key, level):
    if 'LR' in key and 'path' not in key:
        if key == 'LR':
            return True
        else:
            return int(key[3]) < level
    return False


def generate_code_samples(model, data, opt, keep_last_proj=False, print_out=True):
    options = opt['train']

    # For DCI
    dci_num_comp_indices = int(options['dci_num_comp_indices'])
    dci_num_simp_indices = int(options['dci_num_simp_indices'])
    sample_perturbation_magnitude = float(options['sample_perturbation_magnitude'])
    # Block and thread size for parallel CUDA programming
    block_size = 100 if 'block_size' not in options else options['block_size']
    thread_size = 10 if 'thread_size' not in options else options['thread_size']
    # Used for DCI query
    num_outer_iterations = 5000 if 'num_outer_iterations' not in options else options['num_outer_iterations']
    all_num_samples_per_img = options['num_samples_per_img']
    num_levels = opt['levels']

    sampled_codes = []
    sampled_targets = set()

    if print_out:
        print("Generating Samples")
    with torch.no_grad():
        for level_num in range(1, num_levels + 1):
            num_instances = data['network_input'][0].shape[0]
            torch.cuda.empty_cache()
            target_name, target_data = get_data_at_level(data, level_num)
            sampled_targets.add(target_name)
            project_dim = 1000 if 'project_dims' not in options else options['project_dims'][level_num - 1]
            mini_batch_size = 20 if 'mini_batch_size' not in options else options['mini_batch_size']
            num_samples_per_img = all_num_samples_per_img[level_num - 1]
            # handle really large target resolution explicitly due to vRAM constraint
            if target_data.shape[-1] > 256:
                project_dim = 700
                mini_batch_size = 5
            if num_samples_per_img < mini_batch_size:
                mini_batch_size = num_samples_per_img
            model.init_projection(h=target_data.shape[-2], w=target_data.shape[-1], total_dim=project_dim)
            dci_db = DCI(project_dim, dci_num_comp_indices, dci_num_simp_indices, block_size, thread_size)

            cur_sampled_code = model.gen_code(data['network_input'][0].shape[0],
                                              data['network_input'][0].shape[2],
                                              data['network_input'][0].shape[3],
                                              levels=[level_num - 1],
                                              tensor_type=torch.empty)[0]

            for sample_index in range(num_instances):
                if (sample_index + 1) % 10 == 0 and print_out:
                    print_without_newline('\rFinding level %d code: Processed %d out of %d instances' % (
                        level_num, sample_index + 1, num_instances))
                code_pool = model.gen_code(num_samples_per_img,
                                           data['network_input'][0].shape[2],
                                           data['network_input'][0].shape[3],
                                           levels=[level_num - 1])[0]
                feature_pool = []

                for i in range(0, num_samples_per_img, mini_batch_size):
                    cur_data = {key: data[key][sample_index] for key in sampled_targets}

                    # fix the previously sampled code
                    if len(code_pool.shape) > 2:
                        code_samples = [cur_code[sample_index].expand(mini_batch_size, -1, -1, -1)
                                        for cur_code in sampled_codes]
                    else:
                        code_samples = [cur_code[sample_index].expand(mini_batch_size, -1)
                                        for cur_code in sampled_codes]
                    # add the new samples
                    code_samples.append(code_pool[i:i + mini_batch_size])

                    cur_data['network_input'] = []
                    for net_inp in range(len(data['network_input'])):
                        cur_data['network_input'].append(
                            data['network_input'][net_inp][sample_index].expand(mini_batch_size, -1, -1, -1))
                    if 'rarity_masks' in data.keys():
                        cur_data['rarity_masks'] = []
                        for rar_msk in range(len(data['rarity_masks'])):
                            cur_data['rarity_masks'].append(data['rarity_masks'][rar_msk][sample_index])

                    model.feed_data(cur_data, code=code_samples)
                    feature_output = model.get_features(level=(level_num - 1))
                    feature_pool.append(feature_output['gen_feat'])

                feature_pool = torch.cat(feature_pool, dim=0)
                dci_db.add(feature_pool.reshape(num_samples_per_img, -1))
                target_feature = feature_output['real_feat']
                best_sample_idx, _ = dci_db.query(
                    target_feature.reshape(target_feature.shape[0], -1), 1, num_outer_iterations)
                nn_index = int(best_sample_idx[0][0])
                cur_sampled_code[sample_index, :] = code_pool[nn_index, :]
                # clear the db
                dci_db.clear()

            if print_out:
                print_without_newline('\rFinding level %d code: Processed %d out of %d instances\n' % (
                    level_num, num_instances, num_instances))
            sampled_codes.append(cur_sampled_code)
            dci_db.free()

            if not (keep_last_proj and level_num == num_levels):
                model.clear_projection()

        torch.cuda.empty_cache()

        # add sample perturbations
        for i, sample in enumerate(sampled_codes):
            sampled_codes[i] = sample + model.gen_code(data['network_input'][0].shape[0],
                                                       data['network_input'][0].shape[2],
                                                       data['network_input'][0].shape[3],
                                                       levels=[i])[0] * sample_perturbation_magnitude

    return sampled_codes
