import { ImageItem } from '../viewmodel/ImageItem';
import { PageConstants } from '../common/constants/PageConstants';
import { CommonConstants } from '../common/constants/CommonConstants';
import resourceManager from '@ohos.resourceManager';

@Component
export struct HomeTabContentListItem {
  private item: ImageItem = new ImageItem('image1', {} as resourceManager.RawFileDescriptor);

  build() {
    Row() {
      // TODO: 将图片修改为自身
      Image($r('app.media.ic_internet'))
        .height(PageConstants.IMAGE_HEIGHT)
        .width(PageConstants.IMAGE_WIDTH)
        .margin({ left: $r('app.float.item_image_margin_left') })
        .borderRadius($r('app.float.image_border_radius'))
      Column() {
        Column() {
          Text(this.item.getName())
            .fontSize($r('app.float.item_font_size'))
            .margin({
              left: $r('app.float.item_text_margin_left'),
              right: $r('app.float.item_text_margin_right')
            })
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.FULL_PERCENT)
        .justifyContent(FlexAlign.Center)
        .alignItems(HorizontalAlign.Start)

        Divider()
          .strokeWidth(PageConstants.DIVIDER_STROKE_WIDTH)
          .color($r('app.color.divider_color'))
          .margin({
            left: $r('app.float.item_divider_margin_left'),
            right: $r('app.float.item_divider_margin_right')
          })
      }
      .height(CommonConstants.FULL_PERCENT)
      .width(PageConstants.LIST_ITEM_ROW_COLUMN_WIDTH)
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(PageConstants.LIST_ITEM_ROW_HEIGHT)
  }
}