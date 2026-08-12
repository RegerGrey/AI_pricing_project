from conjointkit import generate_design, generate_full_factorial


def test_full_factorial_generation(config):
    profiles = generate_full_factorial(config)
    assert len(profiles) == 12
    assert list(profiles.columns) == ["quality", "color", "price"]


def test_design_has_correct_number_of_tasks(config):
    design = generate_design(config, search_iterations=300)
    assert design.tasks["task_id"].nunique() == 3
    assert len(design.tasks) == 6


def test_no_duplicate_alternatives_inside_task(config):
    design = generate_design(config, search_iterations=300)
    profiles = list(config.attributes)
    assert not design.tasks.duplicated(["task_id", *profiles]).any()
