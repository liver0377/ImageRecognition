import { ImageItem } from '../viewmodel/ImageItem'
import { CommonConstants } from '../common/constants/CommonConstants'
import { PageConstants } from '../common/constants/PageConstants'
import  HomeImageListModel from '../viewmodel/HomeImageListModel'

@Component
export struct HomeTabContentButton {
  @Consume imageList : Array<ImageItem>;     // 双向同步

  build() {
    Column() {
      Button($r('app.string.scan_local_image'), {
        type: ButtonType.Normal,
        stateEffect: true
      })
        .borderRadius($r('app.float.tab_border_radius'))
        .fontSize($r('app.float.button_font_size'))
        .height(PageConstants.TAB_BUTTON_HEIGHT)
        .backgroundColor($r('app.color.button_back_ground_color'))
        .onClick(async () => {
            this.imageList = await HomeImageListModel.getLocalImage();
        })
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(PageConstants.TAB_COLUMN_HEIGHT)
    .justifyContent(FlexAlign.Center)
  }
}
