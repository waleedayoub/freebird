from __future__ import annotations

from pydantic import BaseModel, Field


class SubcategoryInfo(BaseModel):
    object_type: str = Field(alias="objectType", default="")
    object_name: str = Field(alias="objectName", default="")
    bird_std_name: str = Field(alias="birdStdName", default="")
    confidence: float = 0.0


class Keyshot(BaseModel):
    image_url: str = Field(alias="imageUrl", default="")
    message: str = ""
    object_category: str = Field(alias="objectCategory", default="")
    sub_category_name: str = Field(alias="subCategoryName", default="")


class MotionEvent(BaseModel):
    trace_id: str = Field(alias="traceId")
    timestamp: float = 0
    device_name: str = Field(alias="deviceName", default="")
    serial_number: str = Field(alias="serialNumber", default="")
    period: float = 0
    image_url: str = Field(alias="imageUrl", default="")
    video_url: str = Field(alias="videoUrl", default="")
    subcategory_info_list: list[SubcategoryInfo] = Field(
        alias="subcategoryInfoList", default_factory=list
    )
    keyshots: list[Keyshot] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def bird_name(self) -> str:
        for info in self.subcategory_info_list:
            if info.object_type == "bird" and info.object_name:
                return info.object_name
        return ""

    @property
    def bird_latin(self) -> str:
        for info in self.subcategory_info_list:
            if info.object_type == "bird" and info.bird_std_name:
                return info.bird_std_name
        return ""

    @property
    def bird_confidence(self) -> float:
        for info in self.subcategory_info_list:
            if info.object_type == "bird":
                return info.confidence
        return 0.0

    @property
    def keyshot_url(self) -> str:
        if self.keyshots and self.keyshots[0].image_url:
            return self.keyshots[0].image_url
        return self.image_url
