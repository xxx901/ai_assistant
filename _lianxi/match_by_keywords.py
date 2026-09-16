def match_by_keywords(keyword_words,user_input: str) -> str | None:
    """
    遍历 KEYWORD_RULES，检查user_input里有没有出现某个类别下的任意关键词，
    命中就返回对应类别名，一个都没命中就返回None。
    """
    for dic in keyword_words :
        for name,keyword in dic.items() :
            if dic in user_input:
                return name
    return None

"""
def match_by_keywords(keyword_rules: dict, user_input: str) -> str | None:
    for name, keywords in keyword_rules.items():           #用  字典.items()  来分别获取字典的key，value。如果直接for 。。。 in 字典   那返回的只是key
        for keyword in keywords:
            if keyword in user_input:
                return name
    return None
    """