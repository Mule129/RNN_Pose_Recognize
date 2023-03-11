# RNN_Pose_Recognize

작동방법 (Data collet : scr\data\pose_collet.py)
2. 카메라 켜진 후  f(front), y(stay1), u(stay2), r(right), l(left), j(jump), b(back)(학습 안함x) 중 하나 선택

3. g(go) 누르면 동작 저장 시작
4. 스페이스바(stop) 누르면 동작 저장 멈춤

#주의사항 : 멈춘 상태에서 동작 선택해야함
#중간에 나쁜 데이터가 들어간 경우 > 30프레임 단위로 저장하기 때문에 저장 전에 멈추면 이전 저장 이후의 프레임 데이터가 사라짐, 혹시 저장까지 된 경우 일단 저장을 멈춰두고(스페이스바) 
> scr\data\move_data\data_collet_school 폴더 안에 지정한 동작(ex : front) 폴더로 들어가서 맨 마지막 데이터를 삭제하면 됨(ex : 135.npy 삭제)
