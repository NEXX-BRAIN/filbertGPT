from filbertgpt.utils.language import language_dict


def load_language_suffix(lang_code: str):
    return f'Please use {language_dict[lang_code]} to answer this question.'
