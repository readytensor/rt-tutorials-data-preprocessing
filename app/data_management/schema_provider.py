
import json


class BinaryClassificationSchema:
    """Class for reading binary classification schema file"""

    def __init__(self, schema_fpath: str) -> None:
        """
        Initializes a new instance of the `BinaryClassificationSchema` class.

        :param schema_fpath: path to the binary classification schema file
        """
        self.schema = json.load(open(schema_fpath))
        self._numeric_features = self._get_features("NUMERIC", "INT", "REAL")
        self._categorical_features = self._get_features("CATEGORICAL")

    def _get_features(self, *types):
        """
        Returns the feature names of the specified data type.

        :param types: data types of the features
        :return: list of feature names
        """
        fields = self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["predictorFields"]
        return [f["fieldName"] for f in fields if f["dataType"] in types]

    @property
    def id_field(self) -> str:
        """
        Gets the name of the ID field.

        :return: name of the ID field
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["idField"]

    @property
    def target_field(self) -> str:
        """
        Gets the name of the target field.

        :return: name of the target field
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["targetField"]

    @property
    def target_class(self) -> str:
        """
        Gets the name of the target class.

        :return: name of the target class
        """
        return self.schema["inputDatasets"]["binaryClassificationBaseMainInput"]["targetClass"]

    @property
    def numeric_features(self) -> list[str]:
        """
        Gets the names of the numeric features.

        :return: list of numeric feature names
        """
        return self._numeric_features

    @property
    def categorical_features(self) -> list[str]:
        """
        Gets the names of the categorical features.

        :return: list of categorical feature names
        """
        return self._categorical_features

    @property
    def features(self) -> list[str]:
        """
        Gets the names of all the features.

        :return: list of all feature names
        """
        return self.numeric_features + self.categorical_features

    @property
    def all_fields(self) -> list[str]:
        """
        Gets the names of all the fields.

        :return: list of all field names
        """
        return [self.id_field, self.target_field] + self.features


    def __str__(self): 
        return str(self.schema)