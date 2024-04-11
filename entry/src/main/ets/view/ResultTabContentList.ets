import { ResultTabContentListItem } from '../view/ResultTabContentListItem'
import { ResultItem } from '../viewmodel/ResultItem'
import { ImageItem } from '../viewmodel/ImageItem'
import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import router from '@ohos.router';

@Component
export struct ResultTabContentList {
  @Consume resultList : Array<ResultItem>;

  async aboutToAppear() {
    // this.resultList = await HomeTabContentListModel.getLocalImage();

  }

  build() {
    Column() {
      List({
        space: PageConstants.LIST_SPACE,
        initialIndex: PageConstants.LIST_INITIAL_INDEX
      }) {
        ForEach(this.resultList, (result: ResultItem, index?: number) => {
          ListItem() {
            ResultTabContentListItem({ result: result });
          }.onClick(() => {
            // GlobalContext.getContext().setObject('globalVideoList', this.videoList);
            // router.pushUrl({
            //   url : CommonConstants.PAGE,
            //   params: {
            //     fd: item.getFd()
            //   }
            // }).catch((err: Error) => {
            //   // TODO: 异常日志
            //   // Logger.error('[IndexTabLocalList] router error: ' + JSON.stringify(err))
            // });

          })


        }, (item: ImageItem) => JSON.stringify(item))
      }
      .backgroundColor(Color.White)
      .borderRadius($r('app.float.list_border_radius'))
    }
    .width(PageConstants.COLUMN_WIDTH)
    .height(CommonConstants.FULL_PERCENT)
  }
}