import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { ResultTabContentList } from '../view/ResultTabContentList'
import { CommonConstants } from '../common/constants/CommonConstants'
import { HttpHelper } from '../common/utils/HttpUtil'
import  Logger  from '../common/utils/Logger'
import { PageConstants } from '../common/constants/PageConstants'
import { truncateChineseCharacters } from  '../common/utils/StringTool'

@Component
export struct ResultTabContent {
  private myIndex: number;
  @Consume currentIndex: number;
  @Consume imageName: string;
  @Consume httpHelper: HttpHelper;

  @Provide resultList: Array<ResultItem> = [new ResultItem("AAAA", "human", 100, "xxx是...", new ImageItem('image1', "jpg"))];

  async generalRecognize() {
    let resultList : Array<ResultItem> = [];

    try {
      let generalResponseUnitList = await this.httpHelper.generalImageRequest();

      generalResponseUnitList.forEach((item) => {
         // let description: string = item.getDescription() == "" ? "" : item.getDescription().
         // let description: string = item.getDescription() == "" ? "" : truncateChineseCharacters(item.getDescription(), 20);
        // 抛弃description
        resultList.push(new ResultItem(item.getKeyword(), item.getTag(), item.getScore(), '', new ImageItem("whatever", "jpg", item.getImageUrl())))
      })

      // DEBUG
      Logger.info("resultList is assembled", resultList.toString());

    } catch(err) {
      Logger.error("generalImageRequest falied", JSON.stringify(err));
    }


    return resultList
  }

  // 当切换页签时, 该函数是否会重新调用?
  aboutToAppear() {
    // 异步执行
    if (this.currentIndex == this.myIndex) {
      this.generalRecognize()
        .then((resultList: Array<ResultItem>) => {
          this.resultList = resultList;
          Logger.info("recognize success, resultList: ", resultList.toString());
        })
        .catch((resultList: Array<ResultItem>) => {
          this.resultList = resultList
          Logger.info("recognize failed, resultList: ", resultList.toString());
        });
    }

  }

  build() {
    Column() {
      ResultTabContentList();
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}