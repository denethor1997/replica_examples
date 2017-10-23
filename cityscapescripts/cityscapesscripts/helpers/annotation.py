import os
import json
from collections import namedtuple

import datetime
import locale

Point = namedtuple('Point', ['x', 'y'])

class CsObject:
    def __init__(self):
        self.label    = []
        self.polygon  = []
        
        self.id       = -1
        self.deleted  = 0
        self.verified = 0
        self.date     = ""
        self.user     = ""
        self.draw     = True

    def __str__(self):
        ployText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    ployText += '({},{}) '.format(p.x, p.y)
            else:
                ployText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[ 0].x, self.polygon[ 0].y,
                    self.polygon[ 1].x, self.polygon[ 1].y,
                    self.polygon[-2].x, self.polygon[-2].y,
                    self.polygon[-1].x, self.polygon[-1].y)
        else:
            ployText = "none"
        text = "Object: {} - {}".format(self.label, polyText)
        return text

    def fromJsonText(self, jsonText, objId):
        self.id = objId
        self.label = str(jsonText['label'])
        self.ploygon = [Point(p[0], p[1]) for p in jsonText['polygon']]
        if 'detected' in jsonText.keys():
            self.deleted = jsonText['deleted']
        else:
            self.deleted = 0
        if 'verified' in jsonText.keys():
            self.verified = jsonText['verified']
        else:
            self.verified = 1
        if 'user' in jsonText.keys():
            self.user = jsonText['user']
        else:
            self.user = ''
        if 'date' in jsonText.keys():
            self.date = jsonText['date']
        else:
            self.date = ''
        if self.deleted == 1:
            self.draw = False
        else:
            self.draw = True

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['id'] = self.id
        objDict['deleted'] = self.deleted
        objDict['verified'] = self.verified
        objDict['user'] = self.user
        objDict['date'] = self.date
        objDict['polygon'] = []
        for pt in self.polygon:
            objDict['polygon'].append([pt.x, pt.y])

        return objDict

    def updateDate(self):
        try:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'us_us')
        except:
            pass
        self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    def delete(self):
        self.deleted = 1
        self.draw    = False

class Annotation:

    def __init__(self):
        self.imgWidth = 0
        self.imgHeight = 0
        self.objects = []

    def toJson(self):
        return json.dumps(self, default = lambda o: o.__dict__, sort_keys = True, indent = 4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects = []
        for objId, objIn in enumerate(jsonDict['objects']):
            obj = CsObject()
            obj.fromJsonText(objIn, objId)
            self.objects.append(obj)

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)

        return jsonDict

    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())

if __name__ == "__main__":
    obj = CsObject()
    obj.label = 'car'
    obj.polygon.append(Point(0, 0))
    obj.polygon.append(Point(1, 0))
    obj.polygon.append(Point(1, 1))
    obj.polygon.append(Point(0, 1))

    print(obj)
