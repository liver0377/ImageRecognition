import { HomeTabContentListItem } from '../view/HomeTabContentListItem'
import { ImageItem } from '../viewmodel/ImageItem'
import  HomeTabContentListModel  from '../viewmodel/HomeImageListModel'
import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import router from '@ohos.router';
import Logger from '../common/utils/Logger'

@Component
export struct HomeTabContentList {
  @Consume imageList : Array<ImageItem>;

  async aboutToAppear() {
    // this.imageList = await HomeTabContentListModel.getLocalImage();
  }

  build() {
    Column() {
      List({
        space: PageConstants.LIST_SPACE,
        initialIndex: PageConstants.LIST_INITIAL_INDEX
      }) {
        ForEach(this.imageList, (item: ImageItem, index?: number) => {
          ListItem() {
            HomeTabContentListItem({ item: item });
          }.onClick(() => {

            //
            router.pushUrl({
              url : CommonConstants.RESULT_PAGE,
              params: {
                imageName: item.getName()
              }
            }).catch((err: Error) => {
              // TODO: 异常日志
              Logger.error('[IndexTabLocalList] router error: ' + JSON.stringify(err))
            });
          })
        }, (item: ImageItem) => JSON.stringify(item))
      }
      .backgroundColor(Color.White)
      .borderRadius($r('app.float.list_border_radius'))
    }
    .width(PageConstants.COLUMN_WIDTH)
    .height(CommonConstants.NINETY_PERCENT)
  }
}