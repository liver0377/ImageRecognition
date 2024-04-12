import { ResultTabContentListItem } from '../view/ResultTabContentListItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { ImageItem } from '../viewmodel/ImageItem'
import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import router from '@ohos.router';
import Logger from '../common/utils/Logger'

@Component
export struct ResultTabContentList {
  @Consume resultList: Array<ResultItem>;

  build() {
    Column() {
        List({
          space: PageConstants.LIST_SPACE,
          initialIndex: PageConstants.LIST_INITIAL_INDEX
        }) {
          // ForEach(this.resultList, (result: ResultItem) => {
          //   ListItem() {
          //      ResultTabContentListItem({ result: result });
          //   }
          // })
        }
        .backgroundColor(Color.White)
        .borderRadius($r('app.float.list_border_radius'))
      }
      .width(PageConstants.COLUMN_WIDTH)
      .height(CommonConstants.FULL_PERCENT)
    }
 }
