import { ImageItem } from '../viewmodel/ImageItem'
import { ResultItem } from '../viewmodel/ResultItem';
import { PageConstants } from '../common/constants/PageConstants';
import { CommonConstants, RecognitionType } from '../common/constants/CommonConstants';
import resourceManager from '@ohos.resourceManager';

@Component
export struct ResultTabContentListItem {
  private result: ResultItem = new ResultItem("none", RecognitionType.General, 100, "none", new ImageItem("iamge1", "jpg"))

  build() {
    Column() {
      // TODO: 将图片修改为自身

      Row() {
        Column() {
          // 名称
          Text($r('app.string.photo_name'))
            .fontSize($r('app.float.item_font_size'))

          // 类型
          Text($r('app.string.result_type'))
            .fontSize($r('app.float.item_font_size'))

          // 可信度
          Text($r('app.string.reliability'))
            .fontSize($r('app.float.item_font_size'))
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.TWENTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.Start)

        // 结果列
        Column() {
          // 识别名称
          Text(this.result.getName())
            .fontSize($r('app.float.item_font_size'))

          // 物体类型
          Text(this.result.getRecognitionType())
            .fontSize($r('app.float.item_font_size'))

          // 置信度
          Text((this.result.getReliability() * 100).toFixed(2).toString() + '%')
            .fontSize($r('app.float.item_font_size'))

        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.FORTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.Start)

        Column() {
          Image(this.result.getImage().getInternetUrl())
            .borderRadius($r('app.float.image_border_radius'))
        }
        .height(CommonConstants.FULL_PERCENT)
        .width(CommonConstants.FORTY_PERCENT)
        .justifyContent(FlexAlign.SpaceBetween)
        .alignItems(HorizontalAlign.End)


      }
      .height(CommonConstants.FULL_PERCENT)
      .width(CommonConstants.NINETY_PERCENT)

      Divider()
        .strokeWidth(PageConstants.DIVIDER_STROKE_WIDTH)
        .color($r('app.color.divider_color'))
        .margin({
          left: $r('app.float.item_divider_margin_left'),
          right: $r('app.float.item_divider_margin_right')
        })
    }
    .width(CommonConstants.FULL_PERCENT)
    .height(PageConstants.LIST_ITEM_ROW_HEIGHT)
    .justifyContent(FlexAlign.SpaceBetween)
  }
}