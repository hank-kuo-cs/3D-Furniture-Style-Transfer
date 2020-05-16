class FurnitureClass:
    BED = 0
    BOOKSHELF = 1
    CABINET = 2
    CHAIR = 3
    LAMP = 4
    SOFA = 5
    TABLE = 6

    class_enum = {
        'bed': 0,
        'bookshelf': 1,
        'cabinet': 2,
        'chair': 3,
        'lamp': 4,
        'sofa': 5,
        'table': 6
    }

    class_names = list(class_enum)

    @classmethod
    def get_class_id(cls, furniture_str: str) -> int:
        furniture_str = furniture_str.lower()

        if furniture_str not in cls.class_enum:
            raise ValueError('Cannot know \"%s\", ' % furniture_str +
                             'all furniture are bed, bookshelf, cabinet, chair, lamp, sofa, table')

        return cls.class_enum[furniture_str]

    @classmethod
    def get_class_path(cls, class_id=None, class_str=None) -> str:
        if class_id and class_str:
            raise ValueError('You cannot pass "class_id" and "class_str" to the "get_class_path" function')
        if not class_id and not class_str:
            raise ValueError('You must pass either "class_id" or "class_str" to the "get_class_path" function')

        if class_id:
            assert isinstance(class_id, int)
            assert 0 <= class_id < len(cls.class_enum)
        # TODO: concatenate dataset path
        raise NotImplementedError()
