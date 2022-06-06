프로젝트 기간 :	2021.07.30 - 2022.04.30
프로젝트 팀원	: 양문영(컴퓨터소프트웨어 4), 지승하(컴퓨터소프트웨어 4)
지도교수	: 이윤상 교수님
프로젝트 멘토 : 설영호 (엔비디아)
프로젝트 명	: Neural Animation Layering for Synthesizing Martial Arts Movements
프로젝트 내용	

-	프로젝트의 이름
Neural Animation Layering for Synthesizing Martial Arts Movements

-	프로젝트 목적
●	2021 Neural Animation Layering for Synthesizing Martial Arts Movements 논문 이해
●	캐릭터의 현재 frame 동작을 기반으로 다음 frame 동작을 예측하는 모델을 생성합니다.

-	수행내용
1.	bvh viewer 구현
2.	데이터 전처리
3.	모델 구현
4.	모델 학습 및 결과 확인

-	프로젝트의 결과
●	전체적으로 흔들리는 모습이지만, motion은 원하는 결과 도출
●	trajectory를 수정한 것에 대해서도 흔들리지만, 원하는 motion이 나타나는 것을 확인
기대효과 및 개선방향	물리법칙을 위반하고 종종 부자연스러운 동작으로 캐릭터의 모션을 깨뜨리는 게임을, 애니메이터가 작업에 있어 큰 변화 없이 흔히 볼 수 있는 전통적인 혼합 및 계층화 기술의 문제를 극복하는 데 도움이 되는 것을 보여준다.

프로젝트 요약	
- 원시 motion capture data에서 제어 가능한 방식으로 다양한 무술 동작을 생성하는 딥 러닝 프레임워크를 제작한다. 주요 관절의 신호로부터 전신 자세를 예측하는 신경 네트워크인 모션 생성기를 제작하고 학습시킨다. 이후 모션 데이터의 선택된 프레임에서 다른 동작으로 연결하기 위해 일련의 독립 제어 모듈을 활용한다. 이 모듈 각각은 고유한 역할을 수행하며 작업별로 나뉘어 값을 입력받고 미래 모션 궤적을 생성한다.
