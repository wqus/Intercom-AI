import json
from pathlib import Path


def _kb():
    data = json.loads(Path('data/knowledge_base.json').read_text(encoding='utf-8'))
    return {x['id']: x for x in data if isinstance(x, dict) and x.get('id')}


def test_kb006_has_current_school_switch_rule():
    kb = _kb()['kb_006']
    assert 'не нужно из нее выходить' in kb['template_answer'].lower()
    assert 'любое количество школ' in kb['template_answer'].lower()
    assert any('удалите меня из школы' in x.lower() and 'другую' in x.lower() for x in kb['positive_examples'])
    assert any('несколько школ' in x.lower() for x in kb['positive_examples'])


def test_school_exit_and_school_switch_are_not_collapsed():
    kb = _kb()
    assert 'kb_005' in kb and 'kb_006' in kb
    assert any('сменить школу' in x.lower() or 'перейти в другую школу' in x.lower() for x in kb['kb_006']['positive_examples'])
    assert any('сменить школу' in x.lower() or 'перейти в другую школу' in x.lower() for x in kb['kb_005']['negative_phrases'])
    assert any('полностью выйти из школы' in x.lower() for x in kb['kb_006']['negative_phrases'])
