


class HandPosture:
    def __init__(self):
        self.sholder = None
        self.hip = None
        self.chin = None
        self.ear = None
        self.waist = None
        self.left_hand = None
        self.right_hand = None


    def calculate_body_ref_point(self, pose_landmarks):
        lm = pose_landmarks.landmark
        self.sholder = (1 - ((lm[11].y + lm[12].y) / 2))
        self.hip = 1 - ((lm[24].y + lm[23].y) / 2)
        self.chin = 1 - ((((lm[10].y + lm[9].y) / 2) +
                          ((lm[11].y + lm[12].y) / 2)) / 2)
        self.ear = 1 - ((lm[8].y + lm[7].y) / 2)
        self.waist = (self.hip + self.sholder) / 2

        self.left_hand = 1 - ((lm[15].y + lm[19].y + lm[21].y) / 3)
        self.right_hand = 1 - ((lm[16].y + lm[18].y + lm[20].y) / 3)


class HandPosition(HandPosture):
    def __init__(self):
        super().__init__()

    def hand_position(self, hand_ref_point):
        if hand_ref_point >= self.ear:
            return 0
        elif self.ear > hand_ref_point >= self.chin:
            return 1
        elif self.chin > hand_ref_point >= self.sholder*0.8:
            return 2
        elif self.sholder*0.8 > hand_ref_point >= self.waist:
            return 3
        elif hand_ref_point < self.waist:
            return 4

    def get_hand_position(self, dataflow): #pose_landmark
        self.calculate_body_ref_point(dataflow.pose_results)

        lp = self.hand_position(self.left_hand) if dataflow.right_hand is not None else 9
        rp = self.hand_position(self.right_hand) if dataflow.left_hand is not None else 9

        return [rp, lp]

class PredictionAlgorithm:
    def get_result(self, dataflow) -> list:  # return list of stage eg. [1,0] , [0]
        raise NotImplemented

    def transform_dataflow(self, dataflow) -> None:  # if you want to change dataflow
        return

class HandPosition_ATC(PredictionAlgorithm):
    def __init__(self):
        self.handposition = HandPosition()

    def get_result(self,dataflow):
        return self.handposition.get_hand_position(dataflow)