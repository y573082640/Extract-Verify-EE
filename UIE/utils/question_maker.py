import json
class Question_maker:
    def __init__(self, argument2question_path='data/ee/duee/argument2question.json') -> None:
        with open(argument2question_path, 'r') as fp:
            self.argument2question = json.load(fp)

    def get_question_for_argument(self, event_type, role):
        complete_slot_str = event_type + "-" + role
        query_str = self.argument2question.get(complete_slot_str)
        event_type_str = event_type.split("-")[-1]
        if query_str.__contains__("？"):
            query_str_final = query_str
        if query_str == role:
            query_str_final = "找到{}事件中的{}".format(event_type_str, role)
        elif role == "时间":
            query_str_final = "找到{}{}".format(event_type_str, query_str)
        else:
            query_str_final = "找到{}事件中的{},包括{}".format(
                event_type_str, role, query_str)
        return query_str_final
