import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem';
import { PageConstants } from '../common/constants/PageConstants';
import { CommonConstants, RecognitionType } from '../common/constants/CommonConstants';
import resourceManager from '@ohos.resourceManager';

@Component
export struct ResultTabContentListItem {
  private result: ResultItem = new ResultItem("none", RecognitionType.General, 100, "none", new ImageItem("iamge1.jpg", {
  } as resourceManager.RawFileDescriptor))

  build() {
    Column() {
      // TODO: 将图片修改为自身

      Row() {
        Column() {
          // 名称
          Text($r('app.string.photo_name'))
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })

          // 类型
          Text($r('app.string.result_type'))
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })

          // 可信度
          Text($r('app.string.reliability'))
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.THIRTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.Start)

        // 结果列
        Column() {
          // 识别名称
          Text(this.result.getName())
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })

          // 物体类型
          Text(this.result.getRecognitionType())
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })

          // 置信度
          Text(this.result.getReliability().toString() + '%')
            .fontSize($r('app.float.item_font_size'))
            .margin({
              right: $r('app.float.item_text_margin_right')
            })
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.THIRTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.Start)

        Column() {
          Image($r('app.media.ic_internet'))
            .height(PageConstants.IMAGE_HEIGHT)
            .width(PageConstants.IMAGE_WIDTH)
            .borderRadius($r('app.float.image_border_radius'))
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.THIRTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.End)


      }
      .height(CommonConstants.FIFTY_PERCENT)
      .width(CommonConstants.NINETY_PERCENT)
      // .justifyContent(FlexAlign.SpaceAround)

      // 描述信息行
      Row() {
        Text(this.result.getDescription())
          .fontSize($r('app.float.item_font_size'))
          .margin({
            left: $r('app.float.item_text_margin_left'),
            right: $r('app.float.item_text_margin_right')
          })
      }
      .height(CommonConstants.FIFTY_PERCENT)
      .width(CommonConstants.NINETY_PERCENT)
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(PageConstants.LIST_ITEM_ROW_HEIGHT)
    .justifyContent(FlexAlign.SpaceBetween)
  }
}