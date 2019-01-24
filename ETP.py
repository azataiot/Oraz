# Installing some required python packages and models
print("\n**INFO** :Теперь установка некоторых необходимых пакетов и моделей Python\n")

from pyspark.mllib.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.param import Param, Params
from pyspark.sql import SQLContext
print("Done!")
import pyspark
sqlContext = pyspark.SQLContext(pyspark.SparkContext())

print("\n**INFO** :Подготовьте данные обучения из списка (метка, характеристики) кортежей.\n")
# print("Done!")
training = sqlContext.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])
print("Done!")

print("\n**INFO** :# Создать экземпляр LogisticRegression. Этот экземпляр является оценщиком.\n")
# Распечатайте параметры, документацию и любое значение по умолчанию
lr = LogisticRegression(maxIter=10, regParam=0.01)
# распечатайте параметры, документацию и любые значения по умолчанию.
print ("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Изучите модель LogisticRegression. При этом используются параметры, хранящиеся в lr.
model1 = lr.fit(training)

# Так как модель1 является моделью (то есть трансформатором, произведенным оценщиком),
# мы можем просмотреть параметры, которые он использовал во время fit ().
# Это печатает пары параметров (имя: значение), где имена являются уникальными идентификаторами для этого
# Экземпляр LogisticRegression.
print "Model 1 was fit using parameters: "
print model1.extractParamMap()

# Мы можем альтернативно указать параметры, используя словарь Python в качестве paramMap
paramMap = {lr.maxIter: 20}
paramMap[lr.maxIter] = 30 # Укажите 1 Param, перезаписывая исходный maxIter.
paramMap.update({lr.regParam: 0.1, lr.threshold: 0.55}) # Укажите несколько параметров.

#combine paramMaps, которые являются словарями Python.
paramMap2 = {lr.probabilityCol: "myProbability"} # Изменить имя выходного столбца
paramMapCombined = paramMap.copy()
paramMapCombined.update(paramMap2)

# Теперь изучите новую модель, используя параметры paramMapCombined.
# paramMapCombined переопределяет все параметры, установленные ранее с помощью методов lr.set *.
model2 = lr.fit(training, paramMapCombined)
print "Model 2 was fit using parameters: "
print model2.extractParamMap()

# Prepare test data
test = sqlContext.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

# Делайте прогнозы на тестовых данных, используя метод Transformer.transform ().
# LogisticRegression.transform будет использовать только столбец «features».
# Обратите внимание, что model2.transform () выводит столбец «myProbability» вместо обычного
# Столбец «вероятности», поскольку мы переименовали ранее параметр lr.probabilityCol.
prediction = model2.transform(test)
selected = prediction.select("features", "label", "myProbability", "prediction")
for row in selected.collect():
    print row