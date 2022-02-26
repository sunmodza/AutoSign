

class HandFlipFinder:
    def get_result(self,dataflow):
        self.left_flip = 0
        self.right_flip = 0
        if dataflow.left_hand:
            self.left_flip = self.find_handflip(dataflow.left_hand, True)
        if dataflow.right_hand:
            self.right_flip = self.find_handflip(dataflow.right_hand, False)
        return [self.left_flip,self.right_flip]


    def find_handflip(self,hand,is_left):
        landmark = hand.landmark
        index_ref_point = landmark[5]
        pinky_ref_point = landmark[17]
        waist_ref_point = landmark[0]
        front_ref_point = {"x":(index_ref_point.x + pinky_ref_point.x)/2,"y":(index_ref_point.y + pinky_ref_point.y)/2,"z":(index_ref_point.z + pinky_ref_point.z)/2}
        dx = abs(index_ref_point.x - pinky_ref_point.x)
        dy = abs(index_ref_point.y - pinky_ref_point.y)
        dz = abs(index_ref_point.z - pinky_ref_point.z)*2
        dy_waist_ref = abs(((index_ref_point.y + pinky_ref_point.y) / 2) - waist_ref_point.y)
        dz_waist_ref = abs(((index_ref_point.z + pinky_ref_point.z) / 2) - waist_ref_point.z)*2
        dx_waist_ref = abs(((index_ref_point.x + pinky_ref_point.x) / 2) - waist_ref_point.x)

        # print(dx,dy,dz,dx_waist_ref,dy_waist_ref,dz_waist_ref)
        if dy > dx and dy > dz:
            if dx_waist_ref > dz_waist_ref:
                if front_ref_point["x"] > waist_ref_point.x:
                    if index_ref_point.y > pinky_ref_point.y:
                        return 1
                    else:
                        return 2
                else:
                    if index_ref_point.y > pinky_ref_point.y:
                        return 3
                    else:
                        return 4
            else:
                if index_ref_point.y > pinky_ref_point.y:
                    return 5
                else:
                    return 6

        if dx > dy and dx > dz:
            if dy_waist_ref > dz_waist_ref:
                if front_ref_point["y"] > waist_ref_point.y:
                    if index_ref_point.x > pinky_ref_point.x:
                        return 7 if is_left else 8
                    else:
                        return 8 if is_left else 7
                else:
                    if index_ref_point.x > pinky_ref_point.x:
                        return 9 if is_left else 10
                    else:
                        return 10 if is_left else 9
            else:
                if index_ref_point.x > pinky_ref_point.x:
                    return 11 if is_left else 12
                else:
                    return 12 if is_left else 11

        if dz > dy and dz > dx:
            if dy_waist_ref > dx_waist_ref:
                if front_ref_point["y"] > waist_ref_point.y:
                    if index_ref_point.z > pinky_ref_point.z:
                        return 13
                    else:
                        return 14
                else:
                    if index_ref_point.z > pinky_ref_point.z:
                        return 15
                    else:
                        return 16
            else:
                if index_ref_point.z > pinky_ref_point.z:
                    return 17
                else:
                    return 18

class PredictionAlgorithm:
    def get_result(self, dataflow) -> list:  # return list of stage eg. [1,0] , [0]
        raise NotImplemented

    def transform_dataflow(self, dataflow) -> None:  # if you want to change dataflow
        return

class HandFlip_ATC(PredictionAlgorithm):
    def __init__(self):
        self.hand_flip = HandFlipFinder()

    def get_result(self,dataflow):
        return self.hand_flip.get_result(dataflow)