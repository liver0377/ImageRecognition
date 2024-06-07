/**
 * 全局map
 */
export class GlobalContext {
  private constructor() {
  }

  private static instance: GlobalContext;
  private _objects = new Map<string, Object>();

  public static getContext(): GlobalContext {
    if (!GlobalContext.instance) {
      GlobalContext.instance = new GlobalContext();
    }
    return GlobalContext.instance;
  }

  getObject(value: string): Object | undefined {
    return this._objects.get(value);
  }

  setObject(key: string, objectClass: Object): void {
    if (this._objects.has(key)) {
      // 如果key已经存在，将新的对象添加到数组末尾
      (this._objects.get(key) as Object[]).push(objectClass);
    } else {
      // 如果key不存在，将key和对象作为数组的第一个元素存储
      this._objects.set(key, [objectClass]);
    }
  }
}