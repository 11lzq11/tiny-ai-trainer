package com.tinytrainer.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * LOSS vs Step line chart, drawn by hand on Canvas (no chart dependency).
 * Auto-scales to the data range; downsamples to at most one point per
 * horizontal pixel so a 3000-point history stays cheap to draw.
 */
class LossChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    /** (step, loss) points, oldest first. */
    var points: List<Pair<Long, Float>> = emptyList()
        set(value) {
            field = value
            invalidate()
        }

    private val axisPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GRAY
        strokeWidth = 2f
        textSize = 28f
    }
    private val gridPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#E0E0E0")
        strokeWidth = 1f
    }
    private val linePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#7C3AED")
        strokeWidth = 4f
        style = Paint.Style.STROKE
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.DKGRAY
        textSize = 28f
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val padL = 90f
        val padR = 20f
        val padT = 20f
        val padB = 50f
        val plotW = width - padL - padR
        val plotH = height - padT - padB
        if (plotW <= 0 || plotH <= 0) return

        if (points.isEmpty()) {
            canvas.drawText(
                "还没有训练数据",
                padL, height / 2f, textPaint
            )
            return
        }

        val minStep = points.first().first.toFloat()
        val maxStep = points.last().first.toFloat()
        var minLoss = Float.MAX_VALUE
        var maxLoss = Float.MIN_VALUE
        for ((_, loss) in points) {
            if (loss.isNaN()) continue
            minLoss = min(minLoss, loss)
            maxLoss = max(maxLoss, loss)
        }
        if (minLoss == Float.MAX_VALUE) return
        // A little headroom so the line never touches the frame.
        val lossSpan = max(0.1f, maxLoss - minLoss)
        minLoss -= lossSpan * 0.1f
        maxLoss += lossSpan * 0.1f
        val stepSpan = max(1f, maxStep - minStep)

        // Horizontal gridlines + loss labels (4 rows).
        for (i in 0..4) {
            val y = padT + plotH * i / 4f
            canvas.drawLine(padL, y, padL + plotW, y, gridPaint)
            val labelLoss = maxLoss - (maxLoss - minLoss) * i / 4f
            canvas.drawText("%.2f".format(labelLoss), 8f, y + 10f, textPaint)
        }
        // Step labels (first / last).
        canvas.drawText(minStep.toInt().toString(), padL, height - 12f, textPaint)
        val lastLabel = "${maxStep.toInt()}"
        canvas.drawText(
            lastLabel,
            padL + plotW - textPaint.measureText(lastLabel),
            height - 12f,
            textPaint
        )

        // Downsample: at most one point per pixel column.
        val maxPoints = max(2, plotW.toInt())
        val step = max(1, points.size / maxPoints)
        var prevX = -1f
        var prevY = -1f
        var index = 0
        while (index < points.size) {
            val (s, loss) = points[index]
            if (loss.isNaN()) {
                index += step
                continue
            }
            val x = padL + (s - minStep) / stepSpan * plotW
            val y = padT + (maxLoss - loss) / (maxLoss - minLoss) * plotH
            if (prevX >= 0) canvas.drawLine(prevX, prevY, x, y, linePaint)
            prevX = x
            prevY = y
            index += step
        }
        // Always draw the very last point so the live end is exact.
        val (ls, ll) = points.last()
        if (!ll.isNaN()) {
            val x = padL + (ls - minStep) / stepSpan * plotW
            val y = padT + (maxLoss - ll) / (maxLoss - minLoss) * plotH
            if (prevX >= 0 && (x != prevX || y != prevY)) {
                canvas.drawLine(prevX, prevY, x, y, linePaint)
            }
        }
    }
}
