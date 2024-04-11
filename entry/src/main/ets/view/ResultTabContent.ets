import { ResultItem } from '../viewmodel/ResultItem'
import { ResultTabContentList } from '../view/ResultTabContentList'
import { CommonConstants } from '../common/constants/CommonConstants'

@Component
export struct ResultTabContent {
  @Provide imageList: Array<ResultItem> = [];

  build() {
    Column() {
      ResultTabContentList();
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(CommonConstants.FULL_PERCENT)
  }
}