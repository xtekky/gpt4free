# The contents of this file are automatically written by
# tools/generate_schema_wrapper.py. Do not modify directly.
from . import core
from altair.utils import use_signature
from altair.utils.schemapi import Undefined


class MarkMethodMixin(object):
    """A mixin class that defines mark methods"""

    def mark_arc(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                 baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                 color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                 cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                 cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 discreteBandSize=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 fill=Undefined, fillOpacity=Undefined, filled=Undefined, font=Undefined,
                 fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined, height=Undefined,
                 href=Undefined, innerRadius=Undefined, interpolate=Undefined, invalid=Undefined,
                 limit=Undefined, line=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, point=Undefined, radius=Undefined, radius2=Undefined,
                 radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined, size=Undefined,
                 smooth=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 style=Undefined, tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 theta2Offset=Undefined, thetaOffset=Undefined, thickness=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, x2Offset=Undefined,
                 xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined, yOffset=Undefined,
                 **kwds):
        """Set the chart's mark to 'arc'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="arc", **kwds)
        else:
            copy.mark = "arc"
        return copy

    def mark_area(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'area'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="area", **kwds)
        else:
            copy.mark = "area"
        return copy

    def mark_bar(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                 ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                 baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                 color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                 cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                 cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                 cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined, dir=Undefined,
                 discreteBandSize=Undefined, dx=Undefined, dy=Undefined, ellipsis=Undefined,
                 fill=Undefined, fillOpacity=Undefined, filled=Undefined, font=Undefined,
                 fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined, height=Undefined,
                 href=Undefined, innerRadius=Undefined, interpolate=Undefined, invalid=Undefined,
                 limit=Undefined, line=Undefined, lineBreak=Undefined, lineHeight=Undefined,
                 opacity=Undefined, order=Undefined, orient=Undefined, outerRadius=Undefined,
                 padAngle=Undefined, point=Undefined, radius=Undefined, radius2=Undefined,
                 radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined, size=Undefined,
                 smooth=Undefined, stroke=Undefined, strokeCap=Undefined, strokeDash=Undefined,
                 strokeDashOffset=Undefined, strokeJoin=Undefined, strokeMiterLimit=Undefined,
                 strokeOffset=Undefined, strokeOpacity=Undefined, strokeWidth=Undefined,
                 style=Undefined, tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                 theta2Offset=Undefined, thetaOffset=Undefined, thickness=Undefined,
                 timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                 url=Undefined, width=Undefined, x=Undefined, x2=Undefined, x2Offset=Undefined,
                 xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined, yOffset=Undefined,
                 **kwds):
        """Set the chart's mark to 'bar'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="bar", **kwds)
        else:
            copy.mark = "bar"
        return copy

    def mark_image(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                   ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                   baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                   color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                   cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                   cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                   cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                   dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                   ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                   font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                   height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                   invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                   lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                   outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                   radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                   size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                   strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                   strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                   strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                   theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                   thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                   tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                   x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                   yOffset=Undefined, **kwds):
        """Set the chart's mark to 'image'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="image", **kwds)
        else:
            copy.mark = "image"
        return copy

    def mark_line(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'line'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="line", **kwds)
        else:
            copy.mark = "line"
        return copy

    def mark_point(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                   ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                   baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                   color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                   cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                   cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                   cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                   dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                   ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                   font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                   height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                   invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                   lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                   outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                   radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                   size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                   strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                   strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                   strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                   theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                   thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                   tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                   x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                   yOffset=Undefined, **kwds):
        """Set the chart's mark to 'point'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="point", **kwds)
        else:
            copy.mark = "point"
        return copy

    def mark_rect(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'rect'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="rect", **kwds)
        else:
            copy.mark = "rect"
        return copy

    def mark_rule(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'rule'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="rule", **kwds)
        else:
            copy.mark = "rule"
        return copy

    def mark_text(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'text'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="text", **kwds)
        else:
            copy.mark = "text"
        return copy

    def mark_tick(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                  ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                  baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                  color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                  cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                  cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                  cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                  dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                  ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                  font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                  height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                  invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                  lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                  outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                  radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                  size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                  strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                  strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                  strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                  theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                  thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                  tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                  x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                  yOffset=Undefined, **kwds):
        """Set the chart's mark to 'tick'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="tick", **kwds)
        else:
            copy.mark = "tick"
        return copy

    def mark_trail(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                   ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                   baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                   color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                   cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                   cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                   cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                   dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                   ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                   font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                   height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                   invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                   lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                   outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                   radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                   size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                   strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                   strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                   strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                   theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                   thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                   tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                   x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                   yOffset=Undefined, **kwds):
        """Set the chart's mark to 'trail'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="trail", **kwds)
        else:
            copy.mark = "trail"
        return copy

    def mark_circle(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                    ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                    baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                    color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                    cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                    cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                    cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                    dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                    ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                    font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                    height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                    invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                    lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                    outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                    radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                    size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                    strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                    strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                    strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                    theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                    thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                    tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                    x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined,
                    y2Offset=Undefined, yOffset=Undefined, **kwds):
        """Set the chart's mark to 'circle'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="circle", **kwds)
        else:
            copy.mark = "circle"
        return copy

    def mark_square(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                    ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                    baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                    color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                    cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                    cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                    cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                    dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                    ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                    font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                    height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                    invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                    lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                    outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                    radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined, shape=Undefined,
                    size=Undefined, smooth=Undefined, stroke=Undefined, strokeCap=Undefined,
                    strokeDash=Undefined, strokeDashOffset=Undefined, strokeJoin=Undefined,
                    strokeMiterLimit=Undefined, strokeOffset=Undefined, strokeOpacity=Undefined,
                    strokeWidth=Undefined, style=Undefined, tension=Undefined, text=Undefined,
                    theta=Undefined, theta2=Undefined, theta2Offset=Undefined, thetaOffset=Undefined,
                    thickness=Undefined, timeUnitBand=Undefined, timeUnitBandPosition=Undefined,
                    tooltip=Undefined, url=Undefined, width=Undefined, x=Undefined, x2=Undefined,
                    x2Offset=Undefined, xOffset=Undefined, y=Undefined, y2=Undefined,
                    y2Offset=Undefined, yOffset=Undefined, **kwds):
        """Set the chart's mark to 'square'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="square", **kwds)
        else:
            copy.mark = "square"
        return copy

    def mark_geoshape(self, align=Undefined, angle=Undefined, aria=Undefined, ariaRole=Undefined,
                      ariaRoleDescription=Undefined, aspect=Undefined, bandSize=Undefined,
                      baseline=Undefined, binSpacing=Undefined, blend=Undefined, clip=Undefined,
                      color=Undefined, continuousBandSize=Undefined, cornerRadius=Undefined,
                      cornerRadiusBottomLeft=Undefined, cornerRadiusBottomRight=Undefined,
                      cornerRadiusEnd=Undefined, cornerRadiusTopLeft=Undefined,
                      cornerRadiusTopRight=Undefined, cursor=Undefined, description=Undefined,
                      dir=Undefined, discreteBandSize=Undefined, dx=Undefined, dy=Undefined,
                      ellipsis=Undefined, fill=Undefined, fillOpacity=Undefined, filled=Undefined,
                      font=Undefined, fontSize=Undefined, fontStyle=Undefined, fontWeight=Undefined,
                      height=Undefined, href=Undefined, innerRadius=Undefined, interpolate=Undefined,
                      invalid=Undefined, limit=Undefined, line=Undefined, lineBreak=Undefined,
                      lineHeight=Undefined, opacity=Undefined, order=Undefined, orient=Undefined,
                      outerRadius=Undefined, padAngle=Undefined, point=Undefined, radius=Undefined,
                      radius2=Undefined, radius2Offset=Undefined, radiusOffset=Undefined,
                      shape=Undefined, size=Undefined, smooth=Undefined, stroke=Undefined,
                      strokeCap=Undefined, strokeDash=Undefined, strokeDashOffset=Undefined,
                      strokeJoin=Undefined, strokeMiterLimit=Undefined, strokeOffset=Undefined,
                      strokeOpacity=Undefined, strokeWidth=Undefined, style=Undefined,
                      tension=Undefined, text=Undefined, theta=Undefined, theta2=Undefined,
                      theta2Offset=Undefined, thetaOffset=Undefined, thickness=Undefined,
                      timeUnitBand=Undefined, timeUnitBandPosition=Undefined, tooltip=Undefined,
                      url=Undefined, width=Undefined, x=Undefined, x2=Undefined, x2Offset=Undefined,
                      xOffset=Undefined, y=Undefined, y2=Undefined, y2Offset=Undefined,
                      yOffset=Undefined, **kwds):
        """Set the chart's mark to 'geoshape'
    
        For information on additional arguments, see :class:`MarkDef`
        """
        kwds = dict(align=align, angle=angle, aria=aria, ariaRole=ariaRole,
                    ariaRoleDescription=ariaRoleDescription, aspect=aspect, bandSize=bandSize,
                    baseline=baseline, binSpacing=binSpacing, blend=blend, clip=clip, color=color,
                    continuousBandSize=continuousBandSize, cornerRadius=cornerRadius,
                    cornerRadiusBottomLeft=cornerRadiusBottomLeft,
                    cornerRadiusBottomRight=cornerRadiusBottomRight, cornerRadiusEnd=cornerRadiusEnd,
                    cornerRadiusTopLeft=cornerRadiusTopLeft, cornerRadiusTopRight=cornerRadiusTopRight,
                    cursor=cursor, description=description, dir=dir, discreteBandSize=discreteBandSize,
                    dx=dx, dy=dy, ellipsis=ellipsis, fill=fill, fillOpacity=fillOpacity, filled=filled,
                    font=font, fontSize=fontSize, fontStyle=fontStyle, fontWeight=fontWeight,
                    height=height, href=href, innerRadius=innerRadius, interpolate=interpolate,
                    invalid=invalid, limit=limit, line=line, lineBreak=lineBreak, lineHeight=lineHeight,
                    opacity=opacity, order=order, orient=orient, outerRadius=outerRadius,
                    padAngle=padAngle, point=point, radius=radius, radius2=radius2,
                    radius2Offset=radius2Offset, radiusOffset=radiusOffset, shape=shape, size=size,
                    smooth=smooth, stroke=stroke, strokeCap=strokeCap, strokeDash=strokeDash,
                    strokeDashOffset=strokeDashOffset, strokeJoin=strokeJoin,
                    strokeMiterLimit=strokeMiterLimit, strokeOffset=strokeOffset,
                    strokeOpacity=strokeOpacity, strokeWidth=strokeWidth, style=style, tension=tension,
                    text=text, theta=theta, theta2=theta2, theta2Offset=theta2Offset,
                    thetaOffset=thetaOffset, thickness=thickness, timeUnitBand=timeUnitBand,
                    timeUnitBandPosition=timeUnitBandPosition, tooltip=tooltip, url=url, width=width,
                    x=x, x2=x2, x2Offset=x2Offset, xOffset=xOffset, y=y, y2=y2, y2Offset=y2Offset,
                    yOffset=yOffset, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.MarkDef(type="geoshape", **kwds)
        else:
            copy.mark = "geoshape"
        return copy

    def mark_boxplot(self, box=Undefined, clip=Undefined, color=Undefined, extent=Undefined,
                     median=Undefined, opacity=Undefined, orient=Undefined, outliers=Undefined,
                     rule=Undefined, size=Undefined, ticks=Undefined, **kwds):
        """Set the chart's mark to 'boxplot'
    
        For information on additional arguments, see :class:`BoxPlotDef`
        """
        kwds = dict(box=box, clip=clip, color=color, extent=extent, median=median, opacity=opacity,
                    orient=orient, outliers=outliers, rule=rule, size=size, ticks=ticks, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.BoxPlotDef(type="boxplot", **kwds)
        else:
            copy.mark = "boxplot"
        return copy

    def mark_errorbar(self, clip=Undefined, color=Undefined, extent=Undefined, opacity=Undefined,
                      orient=Undefined, rule=Undefined, size=Undefined, thickness=Undefined,
                      ticks=Undefined, **kwds):
        """Set the chart's mark to 'errorbar'
    
        For information on additional arguments, see :class:`ErrorBarDef`
        """
        kwds = dict(clip=clip, color=color, extent=extent, opacity=opacity, orient=orient, rule=rule,
                    size=size, thickness=thickness, ticks=ticks, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.ErrorBarDef(type="errorbar", **kwds)
        else:
            copy.mark = "errorbar"
        return copy

    def mark_errorband(self, band=Undefined, borders=Undefined, clip=Undefined, color=Undefined,
                       extent=Undefined, interpolate=Undefined, opacity=Undefined, orient=Undefined,
                       tension=Undefined, **kwds):
        """Set the chart's mark to 'errorband'
    
        For information on additional arguments, see :class:`ErrorBandDef`
        """
        kwds = dict(band=band, borders=borders, clip=clip, color=color, extent=extent,
                    interpolate=interpolate, opacity=opacity, orient=orient, tension=tension, **kwds)
        copy = self.copy(deep=False)
        if any(val is not Undefined for val in kwds.values()):
            copy.mark = core.ErrorBandDef(type="errorband", **kwds)
        else:
            copy.mark = "errorband"
        return copy


class ConfigMethodMixin(object):
    """A mixin class that defines config methods"""

    @use_signature(core.Config)
    def configure(self, *args, **kwargs):
        copy = self.copy(deep=False)
        copy.config = core.Config(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_arc(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["arc"] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.AreaConfig)
    def configure_area(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["area"] = core.AreaConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axis(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axis"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisBand(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisBand"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisBottom(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisBottom"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisDiscrete(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisDiscrete"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisLeft(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisLeft"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisPoint(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisPoint"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisQuantitative(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisQuantitative"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisRight(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisRight"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisTemporal(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisTemporal"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisTop(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisTop"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisX(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisX"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXBand(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisXBand"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXDiscrete(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisXDiscrete"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXPoint(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisXPoint"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXQuantitative(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisXQuantitative"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisXTemporal(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisXTemporal"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisY(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisY"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYBand(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisYBand"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYDiscrete(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisYDiscrete"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYPoint(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisYPoint"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYQuantitative(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisYQuantitative"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.AxisConfig)
    def configure_axisYTemporal(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["axisYTemporal"] = core.AxisConfig(*args, **kwargs)
        return copy

    @use_signature(core.BarConfig)
    def configure_bar(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["bar"] = core.BarConfig(*args, **kwargs)
        return copy

    @use_signature(core.BoxPlotConfig)
    def configure_boxplot(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["boxplot"] = core.BoxPlotConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_circle(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["circle"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.CompositionConfig)
    def configure_concat(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["concat"] = core.CompositionConfig(*args, **kwargs)
        return copy

    @use_signature(core.ErrorBandConfig)
    def configure_errorband(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["errorband"] = core.ErrorBandConfig(*args, **kwargs)
        return copy

    @use_signature(core.ErrorBarConfig)
    def configure_errorbar(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["errorbar"] = core.ErrorBarConfig(*args, **kwargs)
        return copy

    @use_signature(core.CompositionConfig)
    def configure_facet(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["facet"] = core.CompositionConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_geoshape(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["geoshape"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_header(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["header"] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerColumn(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["headerColumn"] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerFacet(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["headerFacet"] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.HeaderConfig)
    def configure_headerRow(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["headerRow"] = core.HeaderConfig(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_image(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["image"] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.LegendConfig)
    def configure_legend(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["legend"] = core.LegendConfig(*args, **kwargs)
        return copy

    @use_signature(core.LineConfig)
    def configure_line(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["line"] = core.LineConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_mark(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["mark"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_point(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["point"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.ProjectionConfig)
    def configure_projection(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["projection"] = core.ProjectionConfig(*args, **kwargs)
        return copy

    @use_signature(core.RangeConfig)
    def configure_range(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["range"] = core.RangeConfig(*args, **kwargs)
        return copy

    @use_signature(core.RectConfig)
    def configure_rect(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["rect"] = core.RectConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_rule(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["rule"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.ScaleConfig)
    def configure_scale(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["scale"] = core.ScaleConfig(*args, **kwargs)
        return copy

    @use_signature(core.SelectionConfig)
    def configure_selection(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["selection"] = core.SelectionConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_square(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["square"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.MarkConfig)
    def configure_text(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["text"] = core.MarkConfig(*args, **kwargs)
        return copy

    @use_signature(core.TickConfig)
    def configure_tick(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["tick"] = core.TickConfig(*args, **kwargs)
        return copy

    @use_signature(core.TitleConfig)
    def configure_title(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["title"] = core.TitleConfig(*args, **kwargs)
        return copy

    @use_signature(core.LineConfig)
    def configure_trail(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["trail"] = core.LineConfig(*args, **kwargs)
        return copy

    @use_signature(core.ViewConfig)
    def configure_view(self, *args, **kwargs):
        copy = self.copy(deep=['config'])
        if copy.config is Undefined:
            copy.config = core.Config()
        copy.config["view"] = core.ViewConfig(*args, **kwargs)
        return copy