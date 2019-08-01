from dandelion import DataTXT

def replace_dbpedia(string):
    return string.replace("http://dbpedia.org/resource/", "dbr:")

class DandelionAnnotator:
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key
        self.datatxt = DataTXT(app_id=self.app_id, app_key=self.app_key)

    def dandelion_annotation(self, string):
        """
        Gets a string, annotates it, and returns the annotated version with the entities inside
        :param string:
        :return:
        """

        response = self.datatxt.nex(string, include_lod=True)

        annotated_string = string

        shift = 0
        for annotation in response.annotations:
            start = annotation["start"]
            end = annotation["end"]
            print(shift)
            annotated_string = annotated_string[:start + shift] + replace_dbpedia(
                annotation["lod"].dbpedia) + annotated_string[shift + end:]
            print(annotated_string)
            shift = shift + len(replace_dbpedia(annotation["lod"].dbpedia)) - (annotation["end"] - annotation["start"])

        return annotated_string
