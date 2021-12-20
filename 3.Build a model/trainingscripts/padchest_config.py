# Extended from: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py

# remove missing files
missing_files = ["216840111366964012819207061112010307142602253_04-014-084.png",
            "216840111366964012989926673512011074122523403_00-163-058.png",
            "216840111366964012959786098432011033083840143_00-176-115.png",
            "216840111366964012558082906712009327122220177_00-102-064.png",
            "216840111366964012339356563862009072111404053_00-043-192.png",
            "216840111366964013076187734852011291090445391_00-196-188.png",
            "216840111366964012373310883942009117084022290_00-064-025.png",
            "216840111366964012283393834152009033102258826_00-059-087.png",
            "216840111366964012373310883942009170084120009_00-097-074.png",
            "216840111366964012819207061112010315104455352_04-024-184.png",
            "216840111366964012819207061112010306085429121_04-020-102.png"]

pathologies = ['Air Trapping', 'Aortic Atheromatosis', 'Aortic Elongation', 'Atelectasis',
             'Bronchiectasis', 'Cardiomegaly', 'Consolidation', 'Costophrenic Angle Blunting', 'Edema', 'Effusion',
             'Emphysema', 'Fibrosis', 'Flattened Diaphragm', 'Fracture', 'Granuloma', 'Hemidiaphragm Elevation',
             'Hernia', 'Hilar Enlargement', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
             'Pneumonia', 'Pneumothorax', 'Scoliosis', 'Tuberculosis']                    