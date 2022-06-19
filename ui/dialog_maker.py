from PyQt5 import QtCore, QtGui, QtWidgets
import re
from typing import List

class ClassAdder(QtWidgets.QWidget):
    def __init__(self,name,special_return = None,**kwargs):
        super(ClassAdder, self).__init__()
        self.name = name
        self.kwargs = kwargs
        self.stack = QtWidgets.QVBoxLayout()
        self.special_return = special_return
        self.stack.addWidget(QtWidgets.QLabel(text=f'{name}('))
        if len(kwargs) == 0:
            self.stack.addWidget(QtWidgets.QLabel(text=f'{name}()'))
        else:
            self.stack.addWidget(QtWidgets.QLabel(text=f'{name}('))
            for key,value in kwargs.items():
                if isinstance(value,list):
                    combo = QtWidgets.QComboBox()
                    for i in value:
                        combo.addItem(i)
                    value = combo
                    value.setFixedSize(QtCore.QSize(150, 18))

                elif isinstance(value,bool):
                    radio = QtWidgets.QRadioButton(parent=None)
                    # radio.
                    radio.setChecked(value)
                    radio.setAutoExclusive(False)
                    value = radio

                elif isinstance(value,(str)) and re.search("^N",value):
                    # print(value)
                    _,value_type,default_value = value.split(">")
                    # default_value = float(default_value)

                    if value_type == "i":
                        value = QtWidgets.QSpinBox()
                        value.setValue(int(default_value))
                        value.setMaximum(9999999)
                    else:
                        value = QtWidgets.QLineEdit()
                        value.setText(str(default_value))
                        #value.setDecimals(15)

                    #value.setValue(default_value)
                    #value.setMaximum(9999999)
                    value.setFixedSize(QtCore.QSize(150, 18))

                elif isinstance(value,(ClassAdder,ChooseAndDisplayClassAdder,ChooseAndAdd)):
                    pass

                elif isinstance(value,QtWidgets.QTextEdit):
                    value.setFixedSize(QtCore.QSize(89, 18))


                self.kwargs[key] = value
                p = QtWidgets.QHBoxLayout()
                field = QtWidgets.QLabel(text = f'{key} = ')
                p.addWidget(field)
                p.addWidget(value)
                self.stack.addLayout(p)
            self.stack.addWidget(QtWidgets.QLabel(text=f')'))
        # self.stack.setAlignment(QtCore.Qt.AlignJustify)
        # self.stack.setSizeConstraint()
        self.setLayout(self.stack)

    def toPlainText(self):
        # print(self.stack.children())
        text = f'{self.name}('
        for i,(key,value) in enumerate(self.kwargs.items()):
            if isinstance(value,QtWidgets.QComboBox):
                text += f'{key} = "{value.currentText()}"'
            elif isinstance(value,(QtWidgets.QSpinBox,QtWidgets.QDoubleSpinBox)):
                text += f'{key} = {value.value()}'
            elif isinstance(value,(QtWidgets.QTextEdit,ClassAdder,ChooseAndDisplayClassAdder,ChooseAndAdd)):
                text += f'{key} = {value.toPlainText()}'
            elif isinstance(value,QtWidgets.QRadioButton):
                text += f'{key} = {value.isChecked()}'
            elif isinstance(value,QtWidgets.QLineEdit):
                text += f'{key} = {value.text()}'
            if i < len(self.kwargs)-1:
                text+=","
        #print(text)
        text+=")"
        if self.special_return:
            return self.special_return
        return text

class ChooseAndDisplayClassAdder(QtWidgets.QWidget):
    def __init__(self,all_adder:List[ClassAdder],hor = True,subfix = None):
        super(ChooseAndDisplayClassAdder, self).__init__()
        self.all_adder = all_adder
        self.subfix = subfix
        self.layout = QtWidgets.QHBoxLayout() if hor else QtWidgets.QVBoxLayout()
        self.previous_index = 0
        self.choose_class = QtWidgets.QComboBox()
        for adder in all_adder:
            self.choose_class.addItem(adder.name,userData=adder)
        self.choose_class.currentIndexChanged.connect(self.handle_selected_class)
        self.layout.addWidget(self.choose_class)
        self.layout.addWidget(self.all_adder[self.previous_index])
        self.setLayout(self.layout)

    def handle_selected_class(self,i):
        #self.layout.itemAt(self.layout.count() - 1).widget().destroy()
        self.layout.itemAt(self.layout.count() - 1).widget().hide()
        self.layout.removeWidget(self.layout.itemAt(self.layout.count()-1).widget())
        self.layout.addWidget(self.all_adder[i])
        self.all_adder[i].show()
        #self.layout.replaceWidget(self.all_adder[self.previous_index],self.all_adder[i])
        self.previous_index = i

    def toPlainText(self):
        subfix = "" if self.subfix is None else self.subfix
        if self.all_adder[self.previous_index].special_return == "None":
            return self.all_adder[self.previous_index].toPlainText()
        return subfix+self.all_adder[self.previous_index].toPlainText()


class BaseDialog(QtWidgets.QDialog):
    def __init__(self,maincomponent):

        self.compile_chooser = maincomponent

        super().__init__()

        self.setWindowTitle("Compile")

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout = QtWidgets.QVBoxLayout()
        scroll_area.setWidget(self.compile_chooser)
        self.layout.addWidget(scroll_area)
        self.select_button = QtWidgets.QPushButton()
        self.select_button.clicked.connect(lambda: self.clicked_button(self.compile_chooser.toPlainText()))
        self.select_button.setText("new layer")
        self.layout.addWidget(self.select_button)
        self.setLayout(self.layout)
        self.return_value = None

    def clicked_button(self,text):
        self.return_value = text
        self.close()


class FittingModelDialog(QtWidgets.QDialog):
    def __init__(self):


        super().__init__()

        self.setWindowTitle("Training")

        self.layout = QtWidgets.QVBoxLayout()
        self.show_label = QtWidgets.QLabel()

        self.layout.addWidget(self.show_label)
        self.ok = QtWidgets.QPushButton()
        self.ok.clicked.connect(lambda: self.close())
        self.ok.setText("OK")
        self.layout.addWidget(self.ok)
        self.setLayout(self.layout)
        self.return_value = None


class ChooseAndAdd(QtWidgets.QWidget):
    def __init__(self,adder,  btext="Add"):
        super(ChooseAndAdd, self).__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.button = QtWidgets.QPushButton()
        self.all_choosed = QtWidgets.QListWidget()
        self.button.setText(btext)

        self.ask = BaseDialog(ClassAdder("",l=ChooseAndDisplayClassAdder(adder)))
        self.button.clicked.connect(self.add_class)

        self.layout.addWidget(self.all_choosed)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def add_class(self):
        self.ask.exec_()
        v = self.ask.return_value
        if v is None:
            return
        v = v.replace("(","",1)[4:-1]


        self.all_choosed.addItem(v)

    def toPlainText(self):
        text = ""
        for i in range(self.all_choosed.count()):
            text+=f'{self.all_choosed.item(i).text()},'

        return f'[{text[:-1]}]'

def get_metrics():
    auc = ClassAdder("AUC", num_thresshoulds="N>i>200", curve=["ROC", "PR"],
                     summation_method=["interpolation", "minoring", "majoring"],
                     threshoulds="N>f>None", multi_label=False, num_labels="N>f>None", label_weights="N>f>None",
                     from_logits=False)
    accuracy = ClassAdder("Accuracy")
    cat_accu = ClassAdder("CategoricalAccuracy")
    cat_cro = ClassAdder("CategoricalCrossentropy", from_logits=False, label_smoothing="N>f>0")
    cat_hinge = ClassAdder("CategoricalHinge")
    cos_sim = ClassAdder("CosineSimilarity", axis="N>i>-1")
    fn = ClassAdder("FalseNegatives", thresholds="N>f>0.5")
    fp = ClassAdder("FalsePositives", thresholds="N>f>0.5")
    hinge = ClassAdder("Hinge")
    iou = ClassAdder("IoU", num_classes="N>i>1", target_class_ids="N>f>[0]")
    kldiv = ClassAdder("KLDivergence")
    logcosh = ClassAdder("LogCoshError")
    mean = ClassAdder("Mean")
    mean_abs_error = ClassAdder("MeanAbsoluteError")
    mean_abs_pe = ClassAdder("MeanAbsolutePercentageError")
    mean_iou = ClassAdder("MeanIoU")
    mean_metw = ClassAdder("MeanMetricWrapper")
    mean_re = ClassAdder("MeanRelativeError")
    mse = ClassAdder("MeanSquaredError")
    mse_log = ClassAdder("MeanSquaredLogarithmicError")
    mean_tensor = ClassAdder("MeanTensor")
    one_hot_iou = ClassAdder("OneHotIoU")
    OneHotMean_iou = ClassAdder("OneHotMeanIoU")
    poisson = ClassAdder("Poisson")
    precision = ClassAdder("Precision")
    p_at_r = ClassAdder("PrecisionAtRecall")
    rmse = ClassAdder("RootMeanSquaredError")
    sas = ClassAdder("SensitivityAtSpecificity")
    scata = ClassAdder("SparseCategoricalAccuracy")
    scatc = ClassAdder("SparseCategoricalCrossentropy")
    sptkca = ClassAdder("SparseTopKCategoricalAccuracy")
    spas = ClassAdder("SpecificityAtSensitivity")
    sqh = ClassAdder("SquaredHinge")
    topk = ClassAdder("TopKCategoricalAccuracy")
    tn = ClassAdder("TrueNegatives")
    tp = ClassAdder("TruePositives")

    choices = [auc, accuracy, cat_accu, cat_hinge, cat_accu, cat_cro, cos_sim, fn, fp, hinge, iou, kldiv, logcosh,
               mean,
               mean_abs_error, mean_abs_pe, mean_iou, mean_metw, mean_re, mse, mse_log, mean_tensor,
               one_hot_iou, OneHotMean_iou, poisson, precision, p_at_r, rmse, sas, scatc, scata, sptkca, spas, sqh,
               topk, tn, tp]
    return ChooseAndAdd(choices)

def get_callback():
    model_chk_point = ClassAdder("ModelCheckpoint",filepath="N>f>cb/",monitor="N>f>loss",verbose="N>i>0",save_best_only=True,
                                 mode=["auto","min","max"],save_weights_only=False,save_freq="N>f>epoch")
    tensorboard = ClassAdder("TensorBoard",log_dir="N>f>'logs'",
        histogram_freq="N>i>1",
        write_graph=True,
        write_images=True,
        write_steps_per_second=False,
        update_freq="N>i>1",
        profile_batch="N>i>1",
        embeddings_freq="N>i>0",
        embeddings_metadata="N>f>None")
    early_stopping = ClassAdder("EarlyStopping",monitor="N>f>loss",
        min_delta="N>i>0",
        patience="N>i>0",
        verbose="N>i>0",
        mode=["auto","min","max"],
        baseline="N>f>None",
        restore_best_weights = False)

    rdlr = ClassAdder("ReduceLROnPlateau",monitor="N>f>val_loss",
        factor="N>f>0.1",
        patience="N>i>10",
        verbose="N>i>0",
        mode=["auto","min","max"],
        min_delta="N>f>0.0001",
        cooldown="N>f>0",
        min_lr="N>f>0")

    rm = ClassAdder("RemoteMonitor",root="N>f>localhost",
        path="N>f>/publish/epoch/end/",
        field="N>f>data",
        headers="N>f>None",
        send_as_json=False)

    tonnan = ClassAdder("TerminateOnNaN")
    csv_log = ClassAdder("CSVLogger",filename = "N>f>YOUR FILE", separator="N>f>,", append=False)
    progbar = ClassAdder("ProgbarLogger",count_mode=["samples","steps"], stateful_metrics="N>f>None")

    choice = [model_chk_point,tensorboard,early_stopping,rdlr,rm,tonnan,csv_log,progbar]

    return ChooseAndDisplayClassAdder(choice)






class LayerDialog(QtWidgets.QDialog):
    def __init__(self):
        def create_initializer():
            rdn_adder = ClassAdder("RandomNormal", mean="N>f>0", stddev="N>f>1")
            rdu_adder = ClassAdder("RandomUniform", minval="N>f>-0.05", maxval="N>f>0.05")
            trun_adder = ClassAdder("TruncatedNormal", mean="N>f>0", stddev="N>f>0.05")
            zeros_adder = ClassAdder("Zeros")
            ones_adder = ClassAdder("Ones")
            glorot_normal_adder = ClassAdder("GlorotNormal")
            glorot_uniform_adder = ClassAdder("GlorotUniform")
            henormal_adder = ClassAdder("HeNormal")
            heuniform_adder = ClassAdder("HeUniform")
            identity_adder = ClassAdder("Identity", gain="N>f>1.0")
            orthogonal_adder = ClassAdder("Orthogonal", gain="N>f>1.0")
            constant_adder = ClassAdder("Constant", value="N>f>0.0")
            variance_scaling = ClassAdder("VarianceScaling", scale="N>f>1.0",
                                          distribution=["truncated_normal", "untruncated_normal", "uniform"])
            all_init = [rdn_adder, rdu_adder, trun_adder, zeros_adder, ones_adder, glorot_normal_adder,
                        glorot_uniform_adder, henormal_adder, heuniform_adder, identity_adder, orthogonal_adder,
                        constant_adder, variance_scaling]
            return ChooseAndDisplayClassAdder(all_init,subfix="tf.keras.initializers.")

        def create_regulatrizer_adder():
            l1_adder = ClassAdder("l1", l1="N>f>10e-2")
            l2_adder = ClassAdder("l2", l2="N>f>10e-2")
            l1_l2_adder = ClassAdder("l1_l2", l1="N>f>10e-2", l2="N>f>10e-2")

            all_regularizer = [ClassAdder("None", special_return="None"), l1_adder, l2_adder, l1_l2_adder]
            return ChooseAndDisplayClassAdder(all_regularizer,subfix="tf.keras.regularizers.")

        def create_constraints_adder():
            min_max_norm_adder = ClassAdder("MinMaxNorm", min_value="N>f>0.0", max_value="N>f>1.0",
                                            rate="N>f>1.0",
                                            axis="N>i>0")
            nonneg_adder = ClassAdder("NonNeg")
            unit_norm_adder = ClassAdder("UnitNorm", axis="N>i>0")
            radial_adder = ClassAdder("RadialConstraint")

            return ChooseAndDisplayClassAdder(
                [ClassAdder("None", special_return="None"), min_max_norm_adder, nonneg_adder, unit_norm_adder],subfix="tf.keras.constraints.")

        act_list = ["exponential", "elu", "gelu", "hard_sigmoid", "linear", "relu", "selu", "sigmoid",
                    "softmax", "softplus", "softsign", "swish", "tanh"]

        dense_adder = ClassAdder("Dense", units="N>i>0", activation=act_list,
                                 use_bias=True, kernel_initializer=create_initializer(),
                                 bias_initializer=create_initializer(),
                                 kernel_regularizer=create_regulatrizer_adder(),
                                 bias_regularizer=create_regulatrizer_adder(),
                                 activity_regularizer=create_regulatrizer_adder(),
                                 kernel_constraint=create_constraints_adder(),
                                 bias_constraint=create_constraints_adder())
        conv2d_adder = ClassAdder("Conv2D", filters="N>f>0", kernel_size="N>f>(1,1)", strides="N>f>(1,1)",
                                  padding=["valid", "same"], data_format=["channels_last", "channels_first"],
                                  dilation_rate="N>f>(1,1)", groups="N>i>1", activation=act_list, use_bias=True,
                                  kernel_initializer=create_initializer(), bias_initializer=create_initializer(),
                                  kernel_regularizer=create_regulatrizer_adder(),
                                  bias_regularizer=create_regulatrizer_adder(),
                                  activity_regularizer=create_regulatrizer_adder(),
                                  kernel_constraint=create_constraints_adder(),
                                  bias_constraint=create_constraints_adder())
        conv2dtranspose_adder = ClassAdder("Conv2DTranspose", filters="N>f>0", kernel_size="N>f>(1,1)", strides="N>f>(1,1)",
                                  padding=["valid", "same"], data_format=["channels_last", "channels_first"],
                                  dilation_rate="N>f>(1,1)", output_padding="N>f>None", activation=act_list, use_bias=True,
                                  kernel_initializer=create_initializer(), bias_initializer=create_initializer(),
                                  kernel_regularizer=create_regulatrizer_adder(),
                                  bias_regularizer=create_regulatrizer_adder(),
                                  activity_regularizer=create_regulatrizer_adder(),
                                  kernel_constraint=create_constraints_adder(),
                                  bias_constraint=create_constraints_adder())

        conv1dtranspose_adder = ClassAdder("Conv1DTranspose", filters="N>f>0", kernel_size="N>f>1",
                                           strides="N>f>1",
                                           padding=["valid", "same"], data_format=["channels_last", "channels_first"],
                                           dilation_rate="N>f>1", output_padding="N>f>None", activation=act_list,
                                           use_bias=True,
                                           kernel_initializer=create_initializer(),
                                           bias_initializer=create_initializer(),
                                           kernel_regularizer=create_regulatrizer_adder(),
                                           bias_regularizer=create_regulatrizer_adder(),
                                           activity_regularizer=create_regulatrizer_adder(),
                                           kernel_constraint=create_constraints_adder(),
                                           bias_constraint=create_constraints_adder())

        locallyconnected1d_adder = ClassAdder("LocallyConnected1D", filters="N>f>0", kernel_size="N>f>1",
                                              strides="N>f>1",
                                              padding=["valid", "same"],
                                              data_format=["channels_last", "channels_first"],
                                              dilation_rate="N>f>1", groups="N>i>1", activation=act_list, use_bias=True,
                                              kernel_initializer=create_initializer(),
                                              bias_initializer=create_initializer(),
                                              kernel_regularizer=create_regulatrizer_adder(),
                                              bias_regularizer=create_regulatrizer_adder(),
                                              activity_regularizer=create_regulatrizer_adder(),
                                              kernel_constraint=create_constraints_adder(),
                                              bias_constraint=create_constraints_adder(), implementation="N>i>1")
        locallyconnected2d_adder = ClassAdder("LocallyConnected2D", filters="N>f>0", kernel_size="N>f>(1,1)",
                                              strides="N>f>(1,1)",
                                              padding=["valid", "same"],
                                              data_format=["channels_last", "channels_first"],
                                              activation=act_list,
                                              use_bias=True,
                                              kernel_initializer=create_initializer(),
                                              bias_initializer=create_initializer(),
                                              kernel_regularizer=create_regulatrizer_adder(),
                                              bias_regularizer=create_regulatrizer_adder(),
                                              activity_regularizer=create_regulatrizer_adder(),
                                              kernel_constraint=create_constraints_adder(),
                                              bias_constraint=create_constraints_adder(),
                                              implementation="N>i>1")
        conv1d_adder = ClassAdder("Conv1D", filters="N>f>0", kernel_size="N>f>1", strides="N>f>(1,1)",
                                  padding=["valid", "same"],
                                  data_format=["channels_last", "channels_first"],
                                  dilation_rate="N>f>(1,1)", groups="N>i>1", activation=act_list, use_bias=True,
                                  kernel_initializer=create_initializer(),
                                  bias_initializer=create_initializer(),
                                  kernel_regularizer=create_regulatrizer_adder(),
                                  bias_regularizer=create_regulatrizer_adder(),
                                  activity_regularizer=create_regulatrizer_adder(),
                                  kernel_constraint=create_constraints_adder(),
                                  bias_constraint=create_constraints_adder())
        separable_conv1d_adder = ClassAdder("SeparableConv1D", filters="N>f>0", kernel_size="N>f>1",
                                            strides="N>f>(1,1)",
                                            padding=["valid", "same"],
                                            data_format=["channels_last", "channels_first"],
                                            depth_multiplier="N>f>1",
                                            dilation_rate="N>f>(1,1)", groups="N>i>1", activation=act_list,
                                            use_bias=True,
                                            depthwise_initializer=create_initializer(),
                                            pointwise_initializer=create_initializer(),
                                            pointwise_regularizer=create_regulatrizer_adder(),
                                            depthwise_regularizer=create_regulatrizer_adder(),
                                            bias_initializer=create_initializer(),
                                            activity_regularizer=create_regulatrizer_adder(),
                                            depthwise_constraint=create_constraints_adder(),
                                            pointwise_constraint=create_constraints_adder(),
                                            bias_constraint=create_constraints_adder())

        flatten_adder = ClassAdder("Flatten")
        reshape_adder = ClassAdder("Reshape", target_shape="N>f>(1,-1)")
        randomtranslation_adder = ClassAdder("RandomTranslation", height_factor="N>f>1",
                                             width_factor="N>f>1",
                                             fill_mode=["reflect", "constant", "warp", "nearest"],
                                             interpolation=["bilinear'", "nearest"],
                                             fill_value="N>f>0")
        randomcrop_adder = ClassAdder("RandomCrop", height="N>f>300", width="N>f>300")
        randomflip_adder = ClassAdder("RandomFlip", mode=["horizontal", "vertical", "horizontal_and_vertical"])
        randomrotation_adder = ClassAdder("RandomRotation", factor="N>f>1",
                                          fill_mode=["reflect", "constant", "warp", "nearest"],
                                          interpolation=["bilinear", "nearest"],
                                          fill_value="N>f>0")
        randomzoom_adder = ClassAdder("RandomZoom", height_factor="N>f>1",
                                      width_factor="N>f>None",
                                      fill_mode=["reflect", "constant", "warp", "nearest"],
                                      interpolation=["bilinear", "nearest"],
                                      fill_value="N>f>0")
        randomcontrast_adder = ClassAdder("RandomContrast", factor="N>f>1")
        dropout_adder = ClassAdder("Dropout", rate="N>f>0.2")
        normalization_layer = ClassAdder("Normalization", axis="N>i>-1", mean="N>f>0.5", variance="N>f>1")
        spartial_dropout2d = ClassAdder("SpatialDropout2D", rate="N>f>0.2",
                                        data_format=["channels_first", "channels_last"])
        spatial_dropout1d = ClassAdder("SpatialDropout1D", rate="N>f>0.2")
        gaussian_dropout = ClassAdder("GaussianDropout", rate="N>f>0.2")
        gaussian_noise_dropout = ClassAdder("GaussianNoise", stddev="N>f>0.2")

        maxpooling1d_adder = ClassAdder("MaxPooling1D", pool_size="N>i>2", strides="N>i>1",
                                        padding=["valid", "same"],
                                        data_format=["channels_last", "channels_first"])
        avgpooling1d_adder = ClassAdder("AveragePooling1D", pool_size="N>i>2", strides="N>i>1",
                                        padding=["valid", "same"],
                                        data_format=["channels_last", "'channels_first"])
        glpbalmaxpooling1d_adder = ClassAdder("GlobalMaxPooling1D",
                                              keepdims=False,
                                              data_format=["channels_last", "channels_first"])
        glpbalavgpooling1d_adder = ClassAdder("GlobalAveragePooling1D",
                                              data_format=["channels_last", "channels_first"])

        maxpooling2d_adder = ClassAdder("MaxPooling2D", pool_size="N>f>(2,2)", strides="N>f>(1,1)",
                                        padding=["valid", "same"],
                                        data_format=["channels_last", "channels_first"])
        avgpooling2d_adder = ClassAdder("AveragePooling2D", pool_size="N>f>(2,2)", strides="N>f>(1,1)",
                                        padding=["valid", "same"],
                                        data_format=["channels_last", "channels_first"])
        glpbalmaxpooling2d_adder = ClassAdder("GlobalMaxPooling2D",
                                              keepdims=False,
                                              data_format=["channels_last", "channels_first"])
        glpbalavgpooling2d_adder = ClassAdder("GlobalAveragePooling2D",
                                              keepdims=False,
                                              data_format=["channels_last", "channels_first"])

        zeropad1d_adder = ClassAdder("ZeroPadding1D", padding="N>i>1")
        zeropad2d_adder = ClassAdder("ZeroPadding2D", padding="N>f>(1,1)",
                                     data_format=["channels_last", "channels_first"])

        leakyrelu_adder = ClassAdder("LeakyReLU", alpha="N>f>0.3")
        prelu_adder = ClassAdder("PReLU", alpha_initializer=create_initializer(),
                                 alpha_regularizer=create_regulatrizer_adder(),
                                 alpha_constraint=create_constraints_adder(), shared_axes="N>f>None")
        trelu_adder = ClassAdder("ThresholdedReLU", theta="N>f>1.0")



        self.my_add = ChooseAndDisplayClassAdder(
            sorted([dense_adder, separable_conv1d_adder, conv2d_adder, locallyconnected1d_adder
                       , locallyconnected2d_adder, conv1d_adder, flatten_adder
                       , reshape_adder, randomtranslation_adder, randomflip_adder, randomrotation_adder,
                    randomcontrast_adder, randomzoom_adder, randomcrop_adder
                       , dropout_adder, normalization_layer, spatial_dropout1d, spartial_dropout2d, gaussian_dropout,
                    gaussian_noise_dropout, maxpooling1d_adder, avgpooling1d_adder
                       , glpbalmaxpooling1d_adder, glpbalavgpooling1d_adder, zeropad2d_adder, zeropad1d_adder,
                    leakyrelu_adder, prelu_adder, trelu_adder, glpbalmaxpooling2d_adder, glpbalavgpooling2d_adder,
                    maxpooling2d_adder, avgpooling2d_adder,conv2dtranspose_adder,conv1dtranspose_adder], key=lambda a: a.name), hor=False)

        super().__init__()

        self.setWindowTitle("HELLO!")

        scroll_area = QtWidgets.QScrollArea()
        self.layout = QtWidgets.QVBoxLayout()
        scroll_area.setWidget(self.my_add)
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)
        self.select_button = QtWidgets.QPushButton()
        self.select_button.clicked.connect(lambda: self.clicked_button(self.my_add.toPlainText()))
        self.select_button.setText("new layer")
        self.layout.addWidget(self.select_button)
        self.setLayout(self.layout)
        self.return_value = None

    def clicked_button(self,text):
        self.return_value = text
        self.close()

class CompileDialog(QtWidgets.QDialog):
    def __init__(self):
        adadelta_adder = ClassAdder("Adadelta", learning_rate="N>f>0.001", rho="N>f>0.0", epsilon="N>f>0.0")
        sgd_adder = ClassAdder("SGD", learning_rate='N>f>0.01', momentum='N>f>0',
                               nesterov=False)
        adam_adder = ClassAdder("Adam", learning_rate='N>f>0.001', beta_1='N>f>0.9', beta_2='N>f>0.999',
                                epsilon='N>f>1e-07', amsgrad=False)
        rmsprop_adder = ClassAdder("RMSprop", learning_rate='N>f>0.001', rho='N>f>0.9', momentum='N>f>0.0',
                                   epsilon='N>f>1e-7', centered=False)
        adagrad_adder = ClassAdder("Adagrad", learning_rate='N>f>0.001', initial_accumulator='N>f>0.1',
                                   epsilon='N>f>1e-7')
        adamax_adder = ClassAdder("Adamax", learning_rate='N>f>0.001', beta_1='N>f>0.9', beta_2='N>f>0.999',
                                  epsilon='N>f>1e-07')
        nadam_adder = ClassAdder("Nadam", learning_rate='N>f>0.001', beta_1='N>f>0.9', beta_2='N>f>0.999',
                                 epsilon='N>f>1e-07')
        ftrl_adder = ClassAdder("Ftrl", learning_rate='N>f>0.001', learning_rate_power='N>f>-0.5',
                                initial_accumulator_value="N>f>0.1", l1_regularization_strength="N>f>0.0",
                                l2_regularization_strength="N>f>0.0",
                                l2_shrinkage_regularization_strength="N>f>0.0", beta="N>f>0.0")

        mse_adder = ClassAdder("MSE")
        mae_adder = ClassAdder("MAE")
        cross_entropy_adder = ClassAdder("CategoricalCrossentropy")
        kl_adder = ClassAdder("KLDivergence")
        cosim_adder = ClassAdder("CosineSimilarity")
        msl_adder = ClassAdder("MeanSquaredLogarithmicError")
        map_adder = ClassAdder("MeanAbsolutePercentageError")
        huber_adder = ClassAdder("Huber")
        logcosh_adder = ClassAdder("LogCosh")
        poisson_adder = ClassAdder("Poisson")
        hinge_adder = ClassAdder("Hinge")
        sqhinge_adder = ClassAdder("SquaredHinge")
        cat_hinge_adder = ClassAdder("CategoricalHinge")

        self.compile_chooser = ClassAdder("compile", optimizer=ChooseAndDisplayClassAdder(
            [sgd_adder, adadelta_adder, adam_adder, rmsprop_adder, adagrad_adder, adamax_adder, nadam_adder,
             ftrl_adder]),
                                     loss=ChooseAndDisplayClassAdder(
                                         [mse_adder, mae_adder, cross_entropy_adder, kl_adder, cosim_adder, msl_adder,
                                          map_adder, huber_adder, logcosh_adder, poisson_adder, hinge_adder,
                                          cat_hinge_adder, sqhinge_adder]),metrics = get_metrics())

        super().__init__()

        self.setWindowTitle("Compile")

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout = QtWidgets.QVBoxLayout()
        scroll_area.setWidget(self.compile_chooser)
        self.layout.addWidget(scroll_area)
        self.select_button = QtWidgets.QPushButton()
        self.select_button.clicked.connect(lambda: self.clicked_button(self.compile_chooser.toPlainText()))
        self.select_button.setText("new layer")
        self.layout.addWidget(self.select_button)
        self.setLayout(self.layout)
        self.return_value = None

    def clicked_button(self,text):
        self.return_value = text
        self.close()


class InputDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.compile_chooser = QtWidgets.QComboBox()
        self.all_input = {"pose_results": (33, 3), "right_hand": (21, 3), "left_hand": (21, 3),
                          "face": (468, 3), "image": (400, 400, 3),"image_removed_bg": (400,400,3)}

        for i in self.all_input:
            self.compile_chooser.addItem(i)

        self.setWindowTitle("Compile")

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.compile_chooser)
        self.select_button = QtWidgets.QPushButton()
        self.select_button.clicked.connect(lambda: self.clicked_button(self.compile_chooser.currentText()))
        self.select_button.setText("new layer")
        self.layout.addWidget(self.select_button)
        self.setLayout(self.layout)
        self.return_value = None

    def clicked_button(self, text):
        self.return_value = (text,self.all_input[text])
        self.close()