"""
added by TSTM authors

Segmentation configuration for different tasks
Define which elements should be kept in the segmentation mask for each task
"""
 
# Task segmentation configuration dictionary
TASK_SEGMENTATION_CONFIG = {
    # Reach task: manipulator + target marker
    'reach': {
        'has_object': False,
        'object_site': None,
        'include_target': True,
        'target_site': 'target0',
        'additional_sites': None,
        'description': 'Manipulator reaches the target position'
    },
    
    # Push task: manipulator + object + target marker
    'push': {
        'has_object': True,
        'object_site': 'object0',
        'include_target': True,
        'target_site': 'target0',
        'additional_sites': None,
        'description': 'Push the red object to the target position'
    },
    
    # Peg-in-Box task: manipulator + peg + box (box is both target and object)
    'peg_in_box': {
        'has_object': True,
        'object_site': 'object0',  # peg
        'include_target': False,  # box is added via additional_sites
        'target_site': None,
        'additional_sites': ['box_hole'],  # target box
        'description': 'Insert the peg into the box'
    },
    
    # Hammer task: manipulator + nail board + specific nail targets
    'hammer_all': {
        'has_object': False,  # hammer is part of the manipulator
        'object_site': None,
        'include_target': True,
        'target_site': 'nail_goal1',  # default nail 1; set dynamically at runtime
        'additional_sites': ['nail_board'],  # nail board
        'description': 'Hammer the nail'
    },
}


def get_segmentation_config(task_name):
    """
    Get the segmentation configuration for the specified task
    
    Args:
        task_name: task name ('reach', 'push', 'peg_in_box', 'hammer_all')
    
    Returns:
        config: configuration dict
    """
    # Normalize task name
    task_name = task_name.lower().replace('pegbox', 'peg_in_box').replace('hammerall', 'hammer_all')
    
    if task_name in TASK_SEGMENTATION_CONFIG:
        return TASK_SEGMENTATION_CONFIG[task_name].copy()
    else:
        # Default config: keep only the agent
        print(f"Warning: Unknown task '{task_name}', using default config")
        return {
            'has_object': False,
            'object_site': None,
            'include_target': False,
            'target_site': None,
            'additional_sites': None,
            'description': 'Unknown task'
        }


def print_task_configs():
    """Print segmentation configurations for all tasks"""
    print("=" * 70)
    print("Task segmentation configurations")
    print("=" * 70)
    for task_name, config in TASK_SEGMENTATION_CONFIG.items():
        print(f"\nTask: {task_name}")
        print(f"  Description: {config['description']}")
        print(f"  Has object: {config['has_object']}")
        if config['has_object']:
            print(f"    Object site: {config['object_site']}")
        print(f"  Include target: {config['include_target']}")
        if config['include_target']:
            print(f"    Target site: {config['target_site']}")
        if config['additional_sites']:
            print(f"  Additional sites: {config['additional_sites']}")
    print("=" * 70)


if __name__ == '__main__':
    # Test configuration
    print_task_configs()
