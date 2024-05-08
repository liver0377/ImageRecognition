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
  @Consume resultList: Array<ResultItem>;

  // 当切换页签时, 该函数是否会重新调用?
  aboutToAppear() {
  }

  build() {
    Column() {
      ResultTabContentList();
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}