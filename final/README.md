# Final Project - Multi Source Domain Adaptation

Implement M3SDA for 3 source domains and 1 target domain tasks.

DomainNet dataset includes four domains, infograph, quickdraw, real, and sketch.

For more details, please refer to the [pdf](https://github.com/kkeen699/DLCV-spring2019/blob/master/final/DLCV_final1.pdf) to view the slides of final project 1.

## Usage

To extract the feature first,

    python3 feature.py

To train the model,

    python3 train.py $1
where `$1` is the name of target domain. It should be infograph, quickdraw, real, or sketch.

To predict results,

    python3 ./predict.py $1 $2 $3
- `$1` is a string that indicates the name of the **target domain**.
- `$2` is the directory of testing images (e.g. `./final_data/test`).
- `$3` is the path to your output prediction file (e.g. `./test_predict.csv`).


## Results
|Domain Adaptation|inf+qdr+rel<br>->skt|inf+rel+skt<br>->qdr|qdr+rel+skt<br>->inf|inf+qdr+skt<br>->rel
|----------|-------------|--------|--------|-----|
| Accuracy |  28.73% | 6.26% | 17.64% | 54.29%

